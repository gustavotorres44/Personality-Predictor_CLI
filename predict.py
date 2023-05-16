import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.classify import NaiveBayesClassifier

data_set = pd.read_csv("mbti_1.csv")

all_posts = pd.DataFrame()

types = np.unique(np.array(data_set["type"]))
total = data_set.groupby(["type"]).count() * 50

# Organizing data
for j in types:
    temp1 = data_set[data_set["type"] == j]["posts"]
    temp2 = []
    for i in temp1:
        temp2 += i.split("|||")
    temp3 = pd.Series(temp2)
    all_posts[j] = temp3

useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)


def build_bag_of_words_features_filtered(words):
    words = nltk.word_tokenize(words)
    return {word: 1 for word in words if not word in useless_words}


##### Introverted and Extroverted

# Features for the bag of words model
features = []
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna()  # not all the personality types have same number of files
    if "I" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "introvert") for i in temp1]
        ]
    if "E" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "extrovert") for i in temp1]
        ]

split = []
for i in range(16):
    split += [len(features[i]) * 0.8]
split = np.array(split, dtype=int)

train = []
for i in range(16):
    train += features[i][: split[i]]

IntroExtro = NaiveBayesClassifier.train(train)

#### Intution and Sensing

# Features for the bag of words model
features = []
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna()  # not all the personality types have same number of files
    if "N" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "Intuition") for i in temp1]
        ]
    if "E" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "Sensing") for i in temp1]
        ]

train = []
for i in range(16):
    train += features[i][: split[i]]

IntuitionSensing = NaiveBayesClassifier.train(train)

#### Thinking Feeling
# Features for the bag of words model
features = []
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna()  # not all the personality types have same number of files
    if "T" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "Thinking") for i in temp1]
        ]
    if "F" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "Feeling") for i in temp1]
        ]

train = []
for i in range(16):
    train += features[i][: split[i]]

ThinkingFeeling = NaiveBayesClassifier.train(train)

#### Judging Perceiving
# Features for the bag of words model
features = []
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna()  # not all the personality types have same number of files
    if "J" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "Judging") for i in temp1]
        ]
    if "P" in j:
        features += [
            [(build_bag_of_words_features_filtered(i), "Percieving") for i in temp1]
        ]

train = []
for i in range(16):
    train += features[i][: split[i]]

JudgingPercieiving = NaiveBayesClassifier.train(train)
temp = {
    "train": [
        81.12443979837917,
        70.14524215640667,
        80.03456948570128,
        79.79341109742592,
    ],
    "test": [
        58.20469312585358,
        54.46262259027357,
        59.41315234035509,
        54.40549600629061,
    ],
}

results = pd.DataFrame.from_dict(
    temp,
    orient="index",
    columns=[
        "Introvert - Extrovert",
        "Intuition - Sensing",
        "Thinking - Feeling",
        "Judging - Percieiving",
    ],
)


def MBTI(input):
    tokenize = build_bag_of_words_features_filtered(input)
    ie = IntroExtro.classify(tokenize)
    Is = IntuitionSensing.classify(tokenize)
    tf = ThinkingFeeling.classify(tokenize)
    jp = JudgingPercieiving.classify(tokenize)

    mbt = ""

    if ie == "introvert":
        mbt += "I"
    if ie == "extrovert":
        mbt += "E"
    if Is == "Intuition":
        mbt += "N"
    if Is == "Sensing":
        mbt += "S"
    if tf == "Thinking":
        mbt += "T"
    if tf == "Feeling":
        mbt += "F"
    if jp == "Judging":
        mbt += "J"
    if jp == "Percieving":
        mbt += "P"
    return mbt


def tellmemyMBTI(input, name, traasits=[]):
    a = []
    trait1 = pd.DataFrame([0, 0, 0, 0], ["I", "N", "T", "J"], ["count"])
    trait2 = pd.DataFrame([0, 0, 0, 0], ["E", "S", "F", "P"], ["count"])
    for i in input:
        a += [MBTI(i)]
    for i in a:
        for j in ["I", "N", "T", "J"]:
            if j in i:
                trait1.loc[j] += 1
        for j in ["E", "S", "F", "P"]:
            if j in i:
                trait2.loc[j] += 1
    trait1 = trait1.T
    trait1 = trait1 * 100 / len(input)
    trait2 = trait2.T
    trait2 = trait2 * 100 / len(input)

    # Finding the personality
    YourTrait = ""
    for i, j in zip(trait1, trait2):
        temp = max(trait1[i][0], trait2[j][0])
        if trait1[i][0] == temp:
            YourTrait += i
        if trait2[j][0] == temp:
            YourTrait += j
    traasits += [YourTrait]

    # Plotting

    labels = np.array(results.columns)

    intj = trait1.loc["count"]
    ind = np.arange(4)
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, intj, width, color="royalblue")

    esfp = trait2.loc["count"]
    rects2 = ax.bar(ind + width, esfp, width, color="seagreen")

    fig.set_size_inches(10, 7)

    ax.set_xlabel("Finding the MBTI Trait", size=18)
    ax.set_ylabel("Trait Percent (%)", size=18)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 105, step=10))
    ax.set_title("Your Personality is " + YourTrait, size=20)
    plt.grid(True)

    fig.savefig(name + ".png", dpi=200)

    plt.show()
    return traasits


writings = open("charlotte.txt")
writing = writings.readlines()
writing = [line.rstrip() for line in writing]
print(tellmemyMBTI(writing, "Charlotte"))
