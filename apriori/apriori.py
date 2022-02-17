import random
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations

class Apriori:
    def __init__(self):
        self.movie_data = None


    def read(self):
        # read all movie name data
        print("#################### DATA READ ####################")
        all_movies = pd.read_csv("ml-latest-small/movies.csv")
        all_movies = all_movies.set_index("movieId")
        print("Movie name and genre data read successfully!!!")
        # ead user ratings data
        all_ratings = pd.read_csv("ml-latest-small/ratings.csv")
        print("Ratings data read successfully!!!")
        return all_ratings, all_movies


    def sampling(self, all_ratings, alpha=0.6):
        print("#################### DATA SAMPLING ####################")
        unique_user_id = set(all_ratings['userId'])
        removed_user_id = random.sample(unique_user_id, int((1-alpha)*len(unique_user_id)))
        print("Number of unique user ids before sampling: ", len(unique_user_id))
        all_ratings = all_ratings[~all_ratings['userId'].isin(removed_user_id)]
        print("Number of user id to be removed for sampling: ", len(removed_user_id))
        unique_user_id = set(all_ratings['userId'])
        print("Number of unique user ids after sampling: ", len(unique_user_id))
        return all_ratings


    def print_rules(self, rules, all_movies):
        print("#################### ASSOCIATION RULES ####################")
        # print the association rules with movie names
        for i, movie_ids in enumerate(rules):
            length = len(movie_ids)
            print((i+1), end=") ")
            for idx in range(length):
                print(all_movies.loc[movie_ids[idx]]['title'], end="")
                if idx == len(movie_ids) - 2:
                    print(" -> ", end='')
                elif idx != len(movie_ids) - 1:
                    print(" , ", end="")
            print()
            

    def basket_count(self, movie_id):
        m_id = list(movie_id)
        m_intersection = self.movie_data['userGroup'][m_id[0]]
        for movie in m_id[1:]:
            m_intersection = m_intersection.intersection(self.movie_data['userGroup'][movie])
        return len(m_intersection)


    def preprocessing(self, rating_data, min_supp=100):
        print("#################### DATA PREPROCESSING ####################")
        # preprocessing
        start = datetime.now()
        # removing all movies that has rating < 3
        rating_data = rating_data[rating_data['rating']>=3]

        # group movies as per the movieId to combine the user ids into a set to know the users that have rated that movie
        rating_group_data = rating_data.groupby('movieId')['userId'].apply(set).reset_index(name="userGroup")
        rating_group_data['supp'] = rating_group_data['userGroup'].apply(lambda x: len(x))

        # remove the movieIds where the support is less than minSupp
        rating_group_data = rating_group_data[rating_group_data['supp'] >= min_supp].set_index('movieId')

        time_diff = (datetime.now() - start).total_seconds()
        print("Time taken for initial data processing: {} seconds".format(time_diff))

        return rating_group_data


    def apriori(self, rating_group_data, min_supp=100, min_conf=0.95):
        print("#################### APRIORI ####################")
        self.movie_data = rating_group_data.copy()
        movie_combination = set(list(rating_group_data.index))
        assoc_count = 2
        rules = list()
        movie_data_prev = rating_group_data['supp'].copy()

        while len(movie_combination) > 0:
            print("Calculation for {}-itemset!!!".format(assoc_count))
            # create a dataframe containing the associations as per the association count
            group_movies_inter = None
            group_movies_inter = pd.Series(list(combinations(list(movie_combination), assoc_count))).to_frame("movieId")
            print("Initial candidate itemsets of {}-association: {}".format(assoc_count, len(group_movies_inter)))
            # create set of user groups for each association set and find out the count of each association set as per the no. of users rated
            start = datetime.now()
            group_movies_inter['supp'] = group_movies_inter['movieId'].apply(lambda x: self.basket_count(x))
            time_diff = ((datetime.now() - start).total_seconds())/60.0
            print("Time taken to calculate the user basket count: {} minutes".format(time_diff))
            # group_movies_inter['supp'] = group_movies_inter['userGroup'].apply(lambda x: len(x))
            group_movies_inter = group_movies_inter[group_movies_inter['supp'] >= min_supp]
            print("Final Candidate set of {}-associations >= {}(minSup) found: {}".format(assoc_count, min_supp, len(group_movies_inter)))

            # add the association rules into the association rule list
            if len(group_movies_inter) == 0:
                break
            start = datetime.now()
            group_movies_inter = group_movies_inter.reset_index(drop=True).set_index('movieId')
            for movie_id in group_movies_inter.index:
                idx = tuple(sorted(movie_id))
                for mid in list(combinations(idx, assoc_count-1)):
                    if len(mid) == 1:
                        denominator = movie_data_prev.loc[mid]
                    else:
                        denominator = movie_data_prev.loc[[mid]].iloc[0]['supp']
                    conf = group_movies_inter.loc[[movie_id]].iloc[0]['supp'] * 1.0 / denominator
                    if conf >= min_conf:
                        final = list(mid)
                        final.extend(list(set(idx).difference(set(final))))
                        rules.append(tuple(final))
            print("Length of association rules: ", len(rules))
            time_diff = ((datetime.now() - start).total_seconds())/60.0
            print("Time taken to get the association rules with minimum confidence of {}: {} minutes".format(min_conf, time_diff))

            # get the unique movie ids after the association
            movie_combination = set()
            for x in list(group_movies_inter.index):
                movie_combination = movie_combination.union(set(x))
            print("Number of unique movie ids left after {}-association: {}".format(assoc_count, len(movie_combination)))
            # update the previous dataframe of movie Id and their support
            movie_data_prev = None
            movie_data_prev = group_movies_inter[['supp']].copy().reset_index(drop=False)
            movie_data_prev['movieId'] = movie_data_prev['movieId'].apply(lambda x: tuple(sorted(x)))
            movie_data_prev = movie_data_prev.set_index('movieId')
            assoc_count += 1
            print()
        print("\nTotal association rules found: {}".format(len(rules)))
        return rules