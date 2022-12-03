from relevance_ranking import rel_ranking
from reader import Reader
reader = Reader()

from underthesea import text_normalize
from underthesea import word_tokenize
import json
from collections import defaultdict
from collections import Counter

def load_jsonl(data_path):
    with open(data_path, 'r', encoding='utf-8') as jlines:
        data = [json.loads(jline) for jline in jlines]

    return data

def load_data():
    data_path = './resources/wikipedia_20220620_cleaned.jsonl'
    wiki = load_jsonl(data_path)
    data = [(item['title'], item['text']) for item in wiki]
    
    return data

#Using google to find relevant documents
data = load_data()

with open('./resources/test.json') as f:
    questions = json.load(f)

answer_json = defaultdict(list)
for i, item in enumerate(questions):
    # if i == 1: 
        # break
    ids = item['id']
    question = text_normalize(item['question'])
    
    #Find relevant passages from documents
    passages, n_passages_titles = rel_ranking(question,data)
    print(len(passages))
    # Select top 40 paragraphs
    passages = passages[:20+n_passages_titles]
    passages = [(passage[0], p) for passage in passages for p in passage[1].split('.')]
    #Using reading comprehend model (BERT) to extract answer for each passage
    answers = reader.getPredictions(question,passages)  

    #Reranking passages by answer score
    answers = [[passages[i][0], answers[i][0], answers[i][1], passages[i][1]] for i in range(0,len(answers))]
    answers = [a for a in answers if a[1] != '']
    answers.sort(key = lambda x : x[2],reverse=True)  
    
    n_answers = 10 if len(answers) > 10 else len(answers)
    
    
    answer_counts = Counter()
    for j in range(0, n_answers):
        answer_counts.update(answers[j][1])
        answer_json[ids].append((answers[j][1], answers[j][0], answers[j][2]))  
    tmp = f'question {i+1}: {question}' + '\n'
    if len(answer_counts) != 0:
        tmp += 'Final Ans: ' + str(answer_counts.most_common(1)[0][0]) + '\n'
    else:
        tmp += 'Final Ans: ' + 'null'
    
    for j in range(0, n_answers):
        tmp += 'Tokens: ' + str(len(word_tokenize(answers[j][3]))) + ' Ans: ' + answers[j][1] + ' Score: ' + answers[j][2] + ' Title: ' + answers[j][0] + '\n'
    print(tmp)
    
    if i % 200 == 199:
        out_path = f'./output1/results{i+1}.json'
        with open(out_path, 'w') as f:
            json.dump(answer_json, f, ensure_ascii=False)
    

    
# with open('./output/results.txt', 'w') as f:
#     for line in answer_list:
#         f.write(line)
#         f.write('\n')

# with open('./output/results.json', 'w') as f:
#     json.dump(answer_json, f)