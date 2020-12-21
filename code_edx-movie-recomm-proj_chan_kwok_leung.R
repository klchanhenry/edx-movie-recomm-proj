#
# Code objectives and usage summary:-  
# 
# This is a set of R code for movie recommendation project. It will give a set of 
# result by RMSE telling which method can meet the project objective. 
# 
# By running this code, you will 
# 1. install those library needed automatically. ( can possible takes several minutes )
# 2. download the analysis data from internet ( likely takes several minutes )
# 3. run different kind of machine learning method and measure the performance
# by RMSE. 
# 4. some machine learning method is computational intensive, it will occupy most of
# you CPU time and you may feel your computer respond slowly at the period of time.
# ( likely take 10 to 20 minutes depend on different computer )
# 
# Expect result reference:-
# rmse_results is the table store all RMSE result produce by different.
# if the code run and completed normally, you should obtain something very similar
# by calling the variable "rmse_results" like follow.
#   
# > options(pillar.sigfig=6)
# > rmse_results
# # A tibble: 7 x 2
# method                                                RMSE
# <chr>                                                <dbl>
# 1 Project Target                                    0.8649  
# 2 Just the average                                  1.06005 
# 3 Movie Effect Model                                0.942961
# 4 Movie + User Effects Model                        0.864684
# 5 Matrix Factorization using recosystem             0.784608
# 6 Movie + User Effects Model(Validation)            0.865347
# 7 Matrix Factorization using recosystem(Validation) 0.781588
# 
# Hardware reference:-
# The computer tested on this code is a Intel PC running 2 Cores 4 Threads CPU with 16GBytes
# of memory. The PC was manufactured at around 2015. If your platform is something similar,
# it should be reasonable good to run and having similar operation experience. 
# 
# IMPORTANT Pre-caution:-
# 1. This code will download substantial amount of data from internet. Therefore, 
# be sure you have a reasonable fast and free internet connection.
# 2. This code will run computational intensive task. It is highly recommended to
# dedicate a computer running for this instead of running it while you are having 
# some other important task doing at the same time.
# 3. This code may generate temp file by the system itself, base on my experience, 
# if you have more than 5GB system drive free space, it should be working fine.
# 4. It is highly recommended to run this code on a development machine 
# instead of mission critical machine. By running this code, you accept the risk and
# any unexpected consequent. You are free to examine the code line by lines.
# 5. Finally, if you are not sure what this code will do, you are suggested to review
# and understand instead of rush for execute the code.
# 
#


# load some library for tools on data manipulation
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# START Code for download direct from grouplens.org
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
# END Code for download direct from grouplens.org 

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#rm(dl, ratings, movies, test_index, temp, movielens, removed)
#
#keep the movies varable for analysis
rm(dl, ratings, test_index, temp, movielens, removed)


head(edx,5)

# Creating graph for Number of Rating vs Rating Value
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  labs(y="Number of Rating (count)", x = "Rating Value (rating)") +
  ggtitle("Number of Rating vs Rating Value") +
  geom_point() +
  geom_line()


# collect the number of different type of genres from the training set and form a table
# The more the movie being rated, the more genres will be counted. 
genres = c("Action","Adventure","Animation","Children","Comedy","Crime",
           "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
           "Mystery","Romance","Sci-Fi","Thriller","War","Western")
stat_edx_genres <- sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})
sort(stat_edx_genres,decreasing = TRUE)


# collect the number of different type of genres from the movie dataframe and form a table
# movie dataframe contain all the available movies for rating and also the genres of each movie
stat_movies_genres <- sapply(genres, function(g) {
  sum(str_detect(as.data.frame(movies)$genres, g))
})
sort(stat_movies_genres,decreasing = TRUE)

# Prepare graph for Rating Count vs Number of Movie base on particular genres
library(ggrepel)
stat_movies_genres <- unname(stat_movies_genres)
stat_edx_genres <- unname(stat_edx_genres)
df <- data.frame(stat_movies_genres,stat_edx_genres,genres)
df %>% ggplot(aes(x=stat_movies_genres,y=stat_edx_genres)) +
  geom_point() +
  geom_text_repel(aes(label=genres)) + 
  labs(y="Number of Rating on particular genres", x = "Number of movie with particular genres") +
  ggtitle("Rating Count vs Number of Movie base on particular genres") +
  geom_smooth(method='lm')

# Statistic with focus on user and movie
edx %>% group_by(userId) %>%
  summarise(user_rated_count=n()) %>%
  summary()

edx %>% group_by(movieId) %>%
  summarise(movie_rated_count=n()) %>%
  summary()


# Similar to the previous data set splitting, we split once again and named train_set and test_set
# test set will be 10% of Edx data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in validation set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# calculating average of the rating from train_set dataframe
mu_hat <- mean(train_set$rating)
mu_hat

# run the RMSE to compare with the test_set(internal testing) to give insight on performance
just_average_rmse <- RMSE(test_set$rating, mu_hat)
just_average_rmse

# store the RMSE result into table for comparison
rmse_results <- tibble(method = "Project Target", RMSE = 0.86490)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Just the average", RMSE = just_average_rmse))

# This create movie bias by averaging rating per movie and minus the average rating of all 
# different movies.
mu <- mean(train_set$rating)
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# create the prediction by combining average plus per movie bias
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# run the RMSE to compare with the test_set(internal testing) to give insight on performance
movie_effect_rmse <- RMSE(predicted_ratings, test_set$rating)

# store the RMSE result into table for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = movie_effect_rmse ))

# This create user bias by averaging rating per user and minus the average rating of all 
# different users on all movies.
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# create the prediction by combining average plus per user bias and also per movie bias 
# obtain previously
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# run the RMSE to compare with the test_set(internal testing) to give insight on performance
movie_and_user_effect_rmse <- RMSE(predicted_ratings, test_set$rating)

# store the RMSE result into table for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",
                                     RMSE = movie_and_user_effect_rmse ))

# install the required recosystem library
if(!require(recosystem))
  install.packages("recosystem")

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# arrange the train and test data according to recosystem required
train_reco <- with(train_set, data_memory(
  user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test_set, data_memory(
  user_index = userId, item_index = movieId, rating = rating))

# initial the model object
r <- recosystem::Reco()

# tune the parameter base on the training data information, take around 5 - 10 minutes.
opts_tune <- r$tune(train_reco,                                   
                    opts = list(dim      = c(10, 20, 30),
                                costp_l1 = 0,
                                costp_l2 = c(0.01, 0.1),
                                costq_l1 = 0,
                                costq_l2 = c(0.01, 0.1),
                                lrate    = c(0.01, 0.1),         
                                nthread  = 8,
                                niter    = 10))

# train the model which take around 5 - 10 minutes.
r$train(train_reco, opts = c(opts_tune$min,                      
                             niter = 30, nthread = 8)) 

# predict the test set result.
reco_hat <- r$predict(test_reco, out_memory())

# run the RMSE to compare with the test_set(internal testing) to give insight on performance
reco_matrix_factor_rmse <- RMSE(test_set$rating, reco_hat)

# store the RMSE result into table for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix Factorization using recosystem",
                                     RMSE = reco_matrix_factor_rmse ))

#
# This is the final step, we are challenging our model with the validation set
#

# This is Modelling movie effect with user effect

# calculating average of the rating from whole set of training dataframe
mu_edx <- mean(edx$rating)

# calculating the movie bias
movie_avgs_edx <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_edx))

# calculating the user bias
user_avgs_edx <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_edx - b_i))

# create the validation set rating prediction by combining average plus per user 
# bias and also per movie bias
predicted_ratings_validation <- validation %>%
  left_join(movie_avgs_edx, by='movieId') %>%
  left_join(user_avgs_edx, by='userId') %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  pull(pred)

# run the RMSE to compare with the validation set to give insight on performance
movie_and_user_effect_rmse_validation <- RMSE(predicted_ratings_validation, validation$rating)

# store the RMSE result into table for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model(Validation)",
                                     RMSE = movie_and_user_effect_rmse_validation ))

#
# This is the final step, we are challenging our model with the validation set
#

# install the required recosystem library
if(!require(recosystem))
  install.packages("recosystem")

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# arrange the train and validation data according to recosystem required
train_reco_edx <- with(edx, data_memory(
  user_index = userId, item_index = movieId, rating = rating))
test_reco_validation <- with(validation, data_memory(
  user_index = userId, item_index = movieId, rating = rating))

# initial the model object
r_edx <- recosystem::Reco()

# tune the parameter base on the training data information, take around 5 - 10 minutes.
opts_tune_edx <- r_edx$tune(train_reco_edx,                                   
                            opts = list(dim      = c(10, 20, 30),
                                        costp_l1 = 0,
                                        costp_l2 = c(0.01, 0.1),
                                        costq_l1 = 0,
                                        costq_l2 = c(0.01, 0.1),
                                        lrate    = c(0.01, 0.1),         
                                        nthread  = 8,
                                        niter    = 10))

# train the model which take around 5 - 10 minutes.
r_edx$train(train_reco_edx, opts = c(opts_tune_edx$min,                      
                                     niter = 30, nthread = 8)) 

# predict the validation set result by the model.
reco_hat_validation <- r_edx$predict(test_reco_validation, out_memory())

# run the RMSE to compare with the validation set to give insight on performance
reco_matrix_factor_rmse_validation <- RMSE(validation$rating, reco_hat_validation)

# store the RMSE result into table for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix Factorization using recosystem(Validation)",
                                     RMSE = reco_matrix_factor_rmse_validation ))

options(pillar.sigfig=6)
rmse_results
