import jsonlines
from tqdm import tqdm

view_file = open('/data/zhq/abstract_mrc/GAReader/data/1task_train/Task_1_train.jsonl')
cnt = 0

with open('/data/zhq/abstract_mrc/GAReader/data/option_words.txt','a+') as f:
    
    reader = jsonlines.Reader(view_file)
    for instance in tqdm(reader):
        cnt += 1
        option0 = instance['option_0']
        option1 = instance['option_1']
        option2 = instance['option_2']
        option3 = instance['option_3']
        option4 = instance['option_4']
        f.write(option0 +'\n')
        f.write(option1 +'\n')
        f.write(option2 +'\n')
        f.write(option3 +'\n')
        f.write(option4 +'\n')
    print("行数： " + str(cnt))