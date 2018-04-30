# below import lines are for necessary libraries.
# csv library is imported for storing the processed data.
# I've used ElementTree for parsing XML files to reach the tweets.
# I've used tqdm for showing the process of parsing tweets.
# I've used nltk for part of speech tagging. in nltk I've used TweetTokenizer method since it gives more accurate tokens than word_tokenize function.
# I've used pandas for reading csv file and converting it to dataframe.
# I've used scikit-learn library for main classification and k-fold operation.
# linear_model is for logistic regression selection model_selection is for k-fold.
# train_test_split is for splitting data.
# LabelEncoder is for tweet column in dataframe. Since sklearn does not work with text data directly we vectorize it and give it to sklearn.
import os
import xml.etree.ElementTree as ET
import re
import csv
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model,metrics,model_selection
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




males   = [] #for storing male xml file names.
females = [] #for storing female xml file names.

#vocabulary obtained from: http://languagelog.ldc.upenn.edu/nll/?p=13873 and 5 words in each gender from the given sources on homework page.
female_words = ["love", "hun","inlove","mucho","sweetheart","sweetie","girls",\
"super","soo","uber","tonight","excited","overly","tomorrow","ridiculously","yay",\
"excitement","satisfying","cute","baby","sweet","adorable","hes","cutest","babies","puppy",\
"hes"," lil "," aww ","loving","wonderful","amazing","family","husband","thankful","friends",\
"blessed","boyfriend","grateful","lucky","turns","nephew","celebrate","years","birthday",\
"wonderful","whishing","brother","daughter","sister","son","niece","bestie","ily","hacked"\
"bestfriend","boyfriend","babe","besties","xoxo","im so","her new","chocolate","her","she",\
"with my","best friend","shopping","ugh", "sister", "proud of","mom birthday","mombirthday",\
"mombday", "my hair", "myhair", "cant wait", "cantwait", "loves her", "lovesher", "wonderful",\
"am so", "amso", "so happy","sohappy", "lovely","hubby","excited","dress","boy friend","love you",\
"loveyou","muchfun","much fun","cute","cuties","puppies","i miss","imiss","cleaning","clean","wishesshe",\
"wishes she","yummy","likeiton","like it on","likeit","like it","mommy","girls","omg","husband","loved",\
"beloved","myfamily","my family","giveaway","etsy","baking","hubs","knitting","makeup","make up","edward","jewelry","cute", "love", "boyfriend", "mom", "feel"]


male_words = ["nfl","season","baseball","football","fantasy","sports","play","league","playing","player","game","basketball",\
"team","players","coach","state","government","power","country","freedom","thomas","nation","america","political","human",\
"rights","civil","democracy","librty","society","fight","win","fighting","won","battles","defeat","war","battle","enemy",\
"sword","defeated","meet","bands","victory","fucking","fuckers","fuck","fucks","shut","bullshit","sake","bitches","outta",\
"shit","online","gaming","playing","cod","fifa","pc","live","wii","xbox","games","playin","ps","tag","play","add","steam","pay",\
"public","economy","tax","state","cuts","country","income","taxes","government","obama","trump","debt","budget","health",\
"benefits"," vs ","engineering","callofduty","call of duty","match","hislife","his life","win","cod","league","fucking",\
"creed","himself","http:/","worldcup","world cup","loves his","loveshis","president","ps3","youtube","youtube.com","blackops","black ops","shit","my gf","mygf",\
"my girlfriend","metal","fucking","war","album","modernwarfare","modern warfare","dick","teams","championship","arsenal","fucked","ftw","holyshit",\
"holy shit","lebron","thinks he","thinkshe","his mind","hismind",\
"beard","360","halo","wins","shave","fans","my wife","mywife","wisheshe","wishes he",\
"#news","inflation","sci","muslims","afghanistan","forces","gospel","world's","stocks","system", "software", "game", "based", "site"]


def read_truth_file():
    #this function is for reading the truth file and parsing every xml file for genders and stores them at maes[], females[] lists.
    pathname=os.path.join("en","truth.txt")
    file = open(pathname,'r')
    for line in file:
        temp=line.split(":::")
        if(temp[1]=="female"):
            females.append(temp[0])
        if(temp[1]=="male"):
            males.append(temp[0])
    file.close()

def word_categ_vocab(tweet): #function for counting necessary tags that obtained from nltk, and word counts obtained from vocabulary.
    determiner, preposition, pronoun = 0,0,0
    female_words_count = 0
    male_words_count = 0
    tt      = TweetTokenizer()
    tokens  = tt.tokenize(tweet)    #tokenizing tweet.
    tags    = nltk.pos_tag(tokens)  #part of speech tags for tokens.
    for i in tags:
        if(i[1]=='DT' or i[1]=='PDT' or i[1]=='WDT'): #each tag is obtained from nltk.help.upenn_tags() for required ones.
            determiner+=1
        elif(i[1]=='IN' or i[1]=='RP'):
            preposition+=1
        elif(i[1]=='PRP' or i[1]=='PRP$' or i[1]=='WP' or i[1]=='WP$'):
            pronoun+=1
    
    #below two for loops for vocabulary counting.
    for word in female_words:
        if(word in tweet):
            female_words_count+=1
    for word in male_words:
        if(word in tweet):
            male_words_count+=1
    male_bigram_count = 0
    female_bigram_count = 0
    return determiner,preposition,pronoun,male_words_count,female_words_count
def compound_data(): # function is for:creating a csv file for all tweets, extracting each tweet in all files, tagging it, counting words from.....
    #vocabulary and writing it to csv file.
    print("CSV File has been created.")
    csv_file = open("all_tweets.csv",'w')
    fields   = ['tweet', 'label', 'determiner', 'preposition','pronoun',"male_words","female_words"] #header for csv file.
    writer   = csv.writer(csv_file)
    writer.writerow(fields)
    print("Writing the male tweets.")
    for i in tqdm(males):
        xml_file = i+".xml"
        path = os.path.join("en",xml_file)
        tree = ET.parse(path)
        root = tree.getroot()
        for documents in root.findall('documents'):
            for document in documents:
                tweet = document.text
                determiner, preposition, pronoun,malecount,femalecount = word_categ_vocab(tweet)
                line = [tweet.strip().lower(),0,determiner^3,preposition^3,pronoun^3,malecount,femalecount]
                writer.writerow(line)
    print("Male tweets finished.")
    ################################################################
    print("Writing the female tweets.")
    for j in tqdm(females):
        xml_file = j+".xml"
        path = os.path.join("en",xml_file)
        tree = ET.parse(path)
        root = tree.getroot()
        for documents in root.findall('documents'):
            for document in documents:
                tweet = document.text
                determiner, preposition, pronoun,malecount,femalecount = word_categ_vocab(tweet)
                line = [tweet.strip().lower(),1,determiner^4,preposition^4,pronoun^4,malecount^2,femalecount^2]
                # for females I've gave more weight to the parameters since the female data has less feature set than male.
                # One column of feature may cause significant drops in accuracy.
                writer.writerow(line)
    print("Female tweets finished.")

def classify(): #function is for classification and k-fold part.
    df = pd.read_csv("all_tweets.csv") #reading csv via pandas lib.
    vec = LabelEncoder() 
    df['tweet']=vec.fit_transform(df['tweet'].values.astype('U')) #encoding the tweet column as vector since sklearn does not work with string data.
    df = df.sample(frac=1).reset_index(drop=True) #shuffleing the data frame to distribute the gender order randomly. Otherwise in the first 180000 rows would be male... 
    # and last 180000 rows would be female.
    y  = df.label #gender column as label to give target column to sklearn
    X  = df.drop('label', axis=1).drop('tweet', axis=1) #other columns as features.
    log_reg = linear_model.LogisticRegression() #logistic regression decleration.
    scores  = model_selection.cross_val_score(log_reg, X, y, cv=10, verbose=True) #10-fold cross validation with logistic regression.
    print(scores)
    print('average score: %{}'.format(scores.mean()*100)) #accuracy as %.
def main(): 
    read_truth_file()
    compound_data()
    classify()
if __name__ == "__main__":
    main()
