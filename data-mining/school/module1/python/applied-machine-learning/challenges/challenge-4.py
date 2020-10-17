import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("movie_metadata.csv")

# shapes = data.shape
# columns = data.columns

# a = data[(data["director_name"] == "Michael Bay")]
# b = a.sort_values("num_critic_for_reviews", ascending=False)
# # print(b.head(5))


# # Challenge 3
# c = data[(data["gross"] == 67344392)]
# actor = c["actor_1_name"]
# # print(actor)

# # Challenge 4
# film = data[(data["actor_3_name"] == "Omar Sy")]["movie_title"]
# # print(film)

# # challenge 5
# director = data[(data["director_name"] == "Michael Bay")]
# movie = director[5:6]["movie_title"] # Armageddon
# # print(director[5:6]["actor_1_name"])
# print(data)
# print(data.columns)
# top10 = data["gross"]
# print(top10)


movies = data.sort_values("num_critic_for_reviews", ascending=False).head(10)
print(movies)
print(movies.columns)
d = sns.boxplot(x="director_name", y="gross", data=movies)
plt.show()