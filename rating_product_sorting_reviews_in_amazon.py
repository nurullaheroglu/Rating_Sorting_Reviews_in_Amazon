
# import related libraries
import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# read dataset
df = pd.read_csv("datasets/amazon_review.csv")

# Variables:
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# helpful_yes - number of comments found helpful
# total_vote - total votes


def check_summary(dataframe):
    """
        Take a quick summary about the dataset
    """
    print("########################")
    print(dataframe.head(10))
    print("######## COLUMNS #######")
    print(dataframe.columns)
    print("####### DESCRIBE #######")
    print(dataframe.describe().T)
    print("###### TOTAL NULL ######")
    print(dataframe.isnull().sum())
    print("####### TYPES ##########")
    print(dataframe.dtypes)


check_summary(df)

# Calculated the average score of the product
df["overall"].mean()  # 4.587589

# df['reviewTime'].dtype is object, converting type as datetime
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)

# Set the maximum date as the current date
current_date = pd.to_datetime(str(df['reviewTime'].max()))

# Subtract all the days from the current day and only get the number of days
df["day_diff"] = (current_date - df['reviewTime']).dt.days


def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
    """

        Makes a time-weighted calculation using the day_diff variable based on the current time.

    Parameters
    ----------
    dataframe : dataframe
    w1 : int
        w1 is closest to current time
    w2 : int
    w3 : int
    w4 : int
    w5 : int
        w5 is furthest from current time

    Returns
    -------
        Returns the time-weighted score

    """
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100


time_based_weighted_average(df)  # 4.600026


# Let's specify 20 reviews for the product to be displayed on the product detail page

# We generated the helpful_no variable
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score Calculate
    for details about wilson lower bound score check the links below:
        https://medium.com/tech-that-works/wilson-lower-bound-score-and-bayesian-approximation-for-k-star-scale-rating-to-rate-products-c67ec6e30060
        https://gist.github.com/loisaidasam/4e174fb9f56b05ae549b7b5798cc7f90

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# Calculated wilson lower bound using the helpful_yes and helpful_no columns
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Sorted the first 20 rows according to the wilson lower bound variable
df.sort_values("wilson_lower_bound", ascending=False).head(20)

#           reviewerID        asin                          reviewerName       helpful                                         reviewText  overall                                            summary  unixReviewTime reviewTime  day_diff  helpful_yes  total_vote  helpful_no  wilson_lower_bound
# 2031  A12B7ZMXFI6IXY  B007WTAJTO                  Hyoun Kim "Faluzure"  [1952, 2020]  [[ UPDATE - 6/19/2014 ]]So my lovely wife boug...  5.00000  UPDATED - Great w/ Galaxy S4 & Galaxy Tab 4 10...      1367366400 2013-01-05       701         1952        2020          68             0.95754
# 3449   AOEAD7DPLZE53  B007WTAJTO                     NLee the Engineer  [1428, 1505]  I have tested dozens of SDHC and micro-SDHC ca...  5.00000  Top of the class among all (budget-priced) mic...      1348617600 2012-09-26       802         1428        1505          77             0.93652
# 4212   AVBMZZAFEKO58  B007WTAJTO                           SkincareCEO  [1568, 1694]  NOTE:  please read the last update (scroll to ...  1.00000  1 Star reviews - Micro SDXC card unmounts itse...      1375660800 2013-05-08       578         1568        1694         126             0.91214
# 317   A1ZQAQFYSXL5MQ  B007WTAJTO               Amazon Customer "Kelly"    [422, 495]  If your card gets hot enough to be painful, it...  1.00000                                Warning, read this!      1346544000 2012-02-09      1032          422         495          73             0.81858
# 4672  A2DKQQIZ793AV5  B007WTAJTO                               Twister      [45, 49]  Sandisk announcement of the first 128GB micro ...  5.00000  Super high capacity!!!  Excellent price (on Am...      1394150400 2014-07-03       157           45          49           4             0.80811
# 1835  A1J6VSUM80UAF8  B007WTAJTO                           goconfigure      [60, 68]  Bought from BestBuy online the day it was anno...  5.00000                                           I own it      1393545600 2014-02-28       282           60          68           8             0.78465
