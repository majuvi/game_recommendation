import numpy as np
import pandas as pd
import scipy.sparse as sp

gameplay = ['gameplay%d' % i for i in range(1, 57 + 1)]
challenge = ['challenge%d' % i for i in range(1, 42 + 1)]
motivation = ['motivation%d' % i for i in range(1, 41 + 1) if not i in [35, 37]]  # 35, 37 missing
goodgame = ['goodgame%d' % i for i in range(1, 18 + 1)]
experience = ['Experience_%d' % i for i in range(1, 29 + 1) if
              not i in [23, 24, 27, 28, 29]]  # 23, 24, 27, 28, 29 missing
itemids = gameplay + challenge + motivation + goodgame + experience
itemids_old = gameplay[:52] + challenge[:9]


class Data():

	def __init__(self, add_validation=True):
		self.games_covers = pd.read_csv('cache/covers.csv').set_index('gameid')['url']
		self.games_description = pd.read_csv('cache/games_description.csv').set_index('gameid')['name']
		self.items_description = pd.read_csv('cache/items_description.csv').set_index('qkey')['name'].fillna("")
		self.tags_description = pd.read_csv('cache/tags_description.csv').set_index('tagid')['name']
		
		self.survey_games = pd.read_csv('cache/survey_games.csv')
		self.survey_items = pd.read_csv('cache/survey_items.csv')
		self.survey_tags = pd.read_csv('cache/tags.csv')
		
		if add_validation:
			validation_games, validation_items = self._read_validation()
			self.n_validation = len(validation_games['userid'].unique())
			self.survey_games, self.survey_items = self._add_validation(self.survey_games, self.survey_items, validation_games, validation_items)
			

	def _read_validation(self):
		df = pd.read_csv('cache/validation.csv')

		games_cols = ['fav1', 'fav2', 'fav3']
		rows = [(99, userid, gameid)
				for userid, row in df[games_cols].iterrows()
				for gameid in row]
		validation_games = pd.DataFrame(rows, columns=['questionnaireid', 'userid', 'gameid'])

		items_cols = df.columns[4:]
		rows = [(99, userid, question, answer)
				for userid, answers in df[items_cols].iterrows()
				for question, answer in answers.items() if ~np.isnan(answer)]
		validation_items = pd.DataFrame(rows, columns=['questionnaireid', 'userid', 'qkey', 'answer'])

		return(validation_games, validation_items)

	def _add_validation(self, df_games, df_items, df_validation_games, df_validation_items):
		userid_validation = set(df_validation_games['userid']).union(df_validation_items['userid'])
		userid_last = max(df_games['userid'].max(), df_items['userid'].max())
		addtoid = range(userid_last + 1, userid_last + len(userid_validation) + 1)
		addtoid = dict(zip(userid_validation, addtoid))
		df_validation_games['userid'] = df_validation_games['userid'].map(addtoid)
		df_validation_items['userid'] = df_validation_items['userid'].map(addtoid)
		df_games = pd.concat([df_games, df_validation_games], sort=False)
		df_items = pd.concat([df_items, df_validation_items], sort=False)
		return(df_games, df_items)

	def get_game_matrix(self):
		df = self.survey_games
		rowid, rowid_map = np.unique(df['userid'], return_inverse=True)
		colid, colid_map = np.unique(df['gameid'], return_inverse=True)
		n, rows, cols = len(df), len(rowid), len(colid)
		rowid_idx, colid_idx = np.arange(rows)[rowid_map], np.arange(cols)[colid_map]
		games = sp.csr_matrix((np.ones(n, dtype=bool), (rowid_idx, colid_idx)), shape=(rows,cols))
		return rowid, colid, games

	def get_game_matrix_with_features(self, new_items=False):

		df_games = self.survey_games
		df_items = self.survey_items
		df_tags = self.survey_tags
		
		games_userids = np.unique(df_games['userid'])
		games_gameids = np.unique(df_games['gameid'])
		items_userids = np.unique(df_items['userid'])
		items_itemids = np.unique(df_items['qkey']) if new_items else np.array(itemids_old)
		tags_gameids = np.unique(df_tags['gameid'])
		tags_tagids = np.unique(df_tags['tagid'])

		userids = set(games_userids).intersection(set(items_userids))
		gameids = set(games_gameids).intersection(set(tags_gameids))

		df_games = df_games[df_games['userid'].isin(userids) & df_games['gameid'].isin(gameids)]
		games_userids = np.unique(df_games['userid'])
		games_gameids = np.unique(df_games['gameid'])
		print("Players: %d" % len(games_userids))
		print("Games: %d" % len(games_gameids))
		print("Questions: %d" % len(items_itemids))
		print("Tags: %d" % len(tags_tagids))
		
		df_tags = df_tags[df_tags['gameid'].isin(games_gameids)]
		df_items = df_items[df_items['userid'].isin(games_userids) & df_items['qkey'].isin(items_itemids) & ~df_items['answer'].isnull()]
		
		userid_to_idx = dict(zip(games_userids, np.arange(len(games_userids))))
		gameid_to_idx = dict(zip(games_gameids, np.arange(len(games_gameids))))
		itemid_to_idx = dict(zip(items_itemids, np.arange(len(items_itemids))))
		tagid_to_idx = dict(zip(tags_tagids, np.arange(len(tags_tagids))))

		games = sp.csr_matrix((np.ones(len(df_games), dtype=bool),
							   (df_games['userid'].map(userid_to_idx), df_games['gameid'].map(gameid_to_idx))),
							  shape=(len(games_userids), len(games_gameids)))
		items = sp.csr_matrix((df_items['answer'],
							   (df_items['userid'].map(userid_to_idx), df_items['qkey'].map(itemid_to_idx))),
							  shape=(len(games_userids), len(items_itemids)))
		tags = sp.csr_matrix((np.ones(len(df_tags), dtype=bool),
							   (df_tags['gameid'].map(gameid_to_idx), df_tags['tagid'].map(tagid_to_idx))),
							  shape=(len(games_gameids), len(tags_tagids)))

		return (games, items, tags, games_userids, games_gameids, items_itemids, tags_tagids)

