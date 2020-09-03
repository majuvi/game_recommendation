import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import Image, HTML
from ipywidgets import interact, interactive, fixed, interact_manual
from collections import OrderedDict
pd.set_option('display.max_colwidth', -1)


def add_image_tags(path):
    if str.startswith(path, 'http'):
        return ('<img src="%s" width="75"/>' % path)
    else:
        return ('<p>%s</p>' % path)

def fmt_allcols(df):
    return (dict(zip(df.columns, [add_image_tags] * len(df.columns))))

def display_validation(validation_rows, predict, games, gameids, gamepic, gamename, n_recommendations=10, display=True):
    rows = []
    for row_index in validation_rows:
        row = {}
        gameid_played = gameids[games.getrow(row_index).indices]

        p = predict(row_index)
        gameid_similar = gameids[np.argsort(p)[::-1]]
        gameid_similar = [gameid for gameid in gameid_similar if gameid not in gameid_played]
        gameid_similar = gameid_similar[:n_recommendations]

        for i, gameid in enumerate(gameid_played):
            url = gamepic[gameid] if gameid in gamepic else gamename[gameid]
            row['fav_%02d' % (i + 1)] = url

        for i, gameid in enumerate(gameid_similar):
            url = gamepic[gameid] if gameid in gamepic else gamename[gameid]
            row['rec_%02d' % (i + 1)] = url

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.to_html(escape=False, formatters=fmt_allcols(df), na_rep='')
    if display:
        display(HTML(df))
    else:
        return(df)


def build_display(gameids, gamepic, gamename, columns=10):
    game_url = [gamename[gameid] for gameid in gameids] #[gamepic[gameid] if gameid in gamepic else gamename[gameid] for gameid in gameids]
    rows = [game_url[i:i+columns] for i in range(0, len(game_url), columns)]
    df = pd.DataFrame(rows)
    html = df.to_html(escape=False, formatters=fmt_allcols(df))
    return (html)


def display_validation_questions(validation_rows, predict, userids, games, gameids, gamepic, gamename,
                                 questions=None, itemids=None, itemname=None, n_recommendations=5):
    
    validation = []
    for row in validation_rows:

        userid = userids[row]

        # Question answers
        if not questions is None:
            answers = questions.getrow(row)
            userid_itemids = itemids[answers.indices]
            userid_itemname = [itemname[itemid] for itemid in userid_itemids]

        # Games liked
        gameid_played = gameids[games.getrow(row).indices]
        userid_gamename = [gamename[gameid] for gameid in gameid_played]

        # Recommendations 
        scores = predict(row)
        gameid_similar = gameids[np.argsort(scores)[::-1]]
        gameid_similar = [gameid for gameid in gameid_similar if gameid not in gameid_played]
        gameid_similar = gameid_similar[:n_recommendations]
        #print(gamename[gameid_similar].values)
        userid_recs = [gamename[gameid] for gameid in gameid_similar]
        
        validation.append([userid, ",".join(userid_itemname), ",".join(userid_gamename), ",".join(userid_recs)])

    validation = pd.DataFrame(validation, columns=['userid', 'items', 'games', 'recommendations']).set_index('userid')
    return(validation)


def display_online_questions(predict, gameids, gamepic, gamename, itemids, itemname):

    question_widgets = OrderedDict()
    for itemid in itemids:
        question_widgets[itemid] = widgets.IntSlider(min=-2, max=2, step=1, value=0, description=itemname[itemid],
                                                     style={'description_width': '400px'}, layout={'width': '600px'})
    question_widgets['n_recommendations'] = widgets.IntSlider(min=1, max=40, step=1, value=20,
                                                              description='n_recommendations')

    def online_recommendations(**kwargs):
        n_recommendations = kwargs['n_recommendations']
        answers = np.array([kwargs[itemid] for itemid in kwargs if itemid != 'n_recommendations'])
        df = pd.DataFrame({'gameid': gameids, 'score': predict(answers)})
        gameids_similar = list(df.sort_values('score', ascending=False)['gameid'])[:n_recommendations]

        display(HTML("<h4> Recommendations </h4>"))
        d = build_display(gameids_similar, gamepic, gamename)
        display(HTML(d))

    interact_manual(online_recommendations, **question_widgets)


def display_online(predict, gameids, gamepic, gamename):
    def online_recommendations(fav_game1='442', fav_game2='', fav_game3='', fav_game4='', fav_game5='',
                               n_recommendations=(1, 40)):
        gameid_played = []
        for gameid in [fav_game1, fav_game2, fav_game3, fav_game4, fav_game5]:
            try:
                gameid = int(gameid)
                if gameid in gameids:
                    gameid_played.append(gameid)
            except ValueError:
                continue

        # print(gamename[gameids])
        y = np.isin(gameids, gameid_played)
        P = predict(y)

        gameid_similar = gameids[np.argsort(P)[::-1]]
        gameid_similar = [gameid for gameid in gameid_similar if gameid not in gameid_played]
        gameid_similar = gameid_similar[:n_recommendations]

        display(HTML("<h4> Games Liked </h4>"))
        d = build_display(gameid_played, gamepic, gamename)
        display(HTML(d))

        display(HTML("<h4> Recommendations </h4>"))
        d = build_display(gameid_similar, gamepic, gamename)
        display(HTML(d))

    interact_manual(online_recommendations)

