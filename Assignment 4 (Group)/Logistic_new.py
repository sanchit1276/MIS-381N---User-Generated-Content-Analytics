import numpy
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import model_selection
from nltk.tokenize import word_tokenize
import pandas
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


#Assumptions - DEpendent variable is named Binary, Text is the last column and is named text

stemmer = SnowballStemmer("english")
i=0
#Plotting Confusion Matrix
def show_confusion_matrix(C,class_labels=['0','1']):

    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            '%d'%(tn),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            '%d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            '%d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            '%d'%(tp),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'Error: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'Error: %.2f'%(fn / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,' ',
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            ' ',
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()


#Calculate Lift
def calc_Decile(y_pred,y_actual,y_prob,bins=10):
    cols = ['ACTUAL','PROB_POSITIVE','PREDICTED']
    data = [y_actual,y_prob[:,1],y_pred]
    dfa = pandas.DataFrame(dict(zip(cols,data)))

    #Observations where y=1
    total_positive_n = dfa['ACTUAL'].sum()
    #Total Observations
    dfa= dfa.reset_index()
    total_n = dfa.index.size
    natural_positive_prob = total_positive_n/float(total_n)
    dfa = dfa.sort_values(by=['PROB_POSITIVE'], ascending=[False])
    dfa['rank'] = dfa['PROB_POSITIVE'].rank(method='first')
    #Create Bins where First Bin has Observations with the
    #Highest Predicted Probability that y = 1
    dfa['BIN_POSITIVE'] = pandas.qcut(dfa['rank'],bins,labels=False)
    pos_group_dfa = dfa.groupby('BIN_POSITIVE')
    #Percentage of Observations in each Bin where y = 1
    lift_positive = pos_group_dfa['ACTUAL'].sum()/pos_group_dfa['ACTUAL'].count()
    lift_index_positive = lift_positive/natural_positive_prob

    #result1 = result.reset_index()
    #Consolidate Results into Output Dataframe
    lift_df = pandas.DataFrame({'LIFT_POSITIVE':lift_positive,
                               'LIFT_POSITIVE_INDEX':lift_index_positive,
                               'BASELINE_POSITIVE':natural_positive_prob})

    return lift_df


def remove_punctuation(s):
    no_punct = ""
    for letter in s:
        if letter not in string_punctuation:
            no_punct += letter
    return no_punct

#Read file
df = pandas.read_csv('jobs-text-only.csv',encoding="ISO-8859-1")

Text_present = input('Text in the data? Yes/No: ')
if Text_present =='Yes':
    #Read the text column---Last Column (Assumption)
    string_punctuation = '''()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''
    stop = stopwords.words('english')
    df.iloc[ :, -1] = df.iloc[ :, -1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    for row in df['text']:
        df.iloc[ i, -1] = remove_punctuation(row)
        i=i+1
    df['text'] = df['text'].str.replace("!"," !")
    df['text'] = df['text'].apply(word_tokenize)
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['text'] = df['text'].apply(lambda x : " ".join(x))
    Text_Column = df.iloc[ :, -1:]
    #Get TFIDF Scores
    sklearn_tfidf = TfidfVectorizer(min_df=.01, max_df =.95, stop_words="english",use_idf=True, smooth_idf=False, sublinear_tf=True)
    sklearn_representation = sklearn_tfidf.fit_transform(Text_Column.iloc[:, 0].tolist())
    Tfidf_Output = pandas.DataFrame(sklearn_representation.toarray(), columns=sklearn_tfidf.get_feature_names())

    #Append the column to the final dataset
    Input = pandas.concat([df, Tfidf_Output], axis=1)
    Input = Input.drop('text', 1)
else:
    Input = df

#split into training: 60 and testing set: 40
# train, test = sklearn.cross_validation.train_test_split(Input, train_size = 0.6,random_state=1)
X = Input.loc[:, Input.columns != 'binary']
Y = Input['binary']
# X_test = test.loc[:, Input.columns != 'binary']
# Y_test = test['binary']

#Logit Regression
classifier = LogisticRegression()
#classifier.fit(X, Y)
# Y_pred = classifier.predict(X_test)

Y_pred = model_selection.cross_val_predict(classifier, X, Y, cv=5)

# Confusion matrix
confusion_matrix = confusion_matrix((numpy.array(Y)), Y_pred)


#Validation score table
#y_prob=classifier.predict_proba(X_test)
from sklearn.model_selection import cross_val_predict
y_prob = cross_val_predict(classifier, X, Y, cv=5, method='predict_proba')

validation_columns = ['Predicted_Probability','Y','Y_pred']
validation_data = [y_prob[:,1],Y,Y_pred]
Validation_LR = pandas.DataFrame(dict(zip(validation_columns,validation_data)))
Validation_LR = pandas.concat([Validation_LR, X], axis=1)
Validation_LR = Validation_LR.sort_values(by=['Predicted_Probability'], ascending=[False])



#Decile chart
Decile_Chart = calc_Decile(Y_pred,Y,y_prob)
Decile_Chart['Bin']=abs(10-Decile_Chart.index)
Decile_Chart = Decile_Chart.sort_values(by=['Bin'], ascending=[False])

plt.subplot(221)
plt.bar(Decile_Chart['Bin'], Decile_Chart['LIFT_POSITIVE_INDEX'], align='center',)
plt.xlabel('Bins')
plt.title('Decile Chart')
plt.xticks(Decile_Chart['Bin'])

# Lift chart
cols = ['ACTUAL','PROB_POSITIVE','PREDICTED']
data = [Y,y_prob[:,1],Y_pred]
Lift_data = pandas.DataFrame(dict(zip(cols,data)))
Lift_data = Lift_data.sort_values(by=['PROB_POSITIVE'], ascending=[False])
Lift_data['cum_actual'] = Lift_data.ACTUAL.cumsum()

Lift_data = Lift_data.reset_index()
del Lift_data ['index']
p = Lift_data['cum_actual']
d = Lift_data.index+1

plt.subplot(222)
plt.plot(d,p,color='blue',marker='o',markersize=.2)
total_positive_n = Lift_data['ACTUAL'].sum()
total_positive_count = Lift_data['ACTUAL'].count()
plt.plot([1,total_positive_count],[1,total_positive_n],color='red',marker='o')

plt.legend(['Cumulative 1 when sorted using predicted values'])
plt.title("Lift Chart")
plt.xlabel("#Cases")
plt.grid()
plt.savefig('Decile_Lift.png')

show_confusion_matrix(confusion_matrix, ['0', '1'])
plt.show()
plt.savefig('Confusion.png')

Validation_LR = Validation_LR.rename(columns={'Predicted_Probability': 'Prob of 1', 'Y_pred': 'Predicted', 'Y': 'Actual'})
Validation_LR.to_csv('Validation_LR.csv', index_label=[Validation_LR.columns.name], index=False)



