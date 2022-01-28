import os, sys
#import os.system
#import os.spawn*
import subprocess
from subprocess import PIPE
import time

def showImg(origImg_p, countexImg_p, att_p, q, ans1, ans2): 
    """
    #test
    proc = subprocess.run("date", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    date = proc.stdout
    print('STDOUT: {}'.format(date))
    """

    #show CountEx images
    subprocess.run(f"cat {q}", shell=True)
    #subprocess.run(f"cat {ans1}", shell=True)
    print("original image")
    subprocess.run(f"img2sixel {origImg_p}", shell=True)
    #subprocess.run(f"cat {ans2}", shell=True)
    print("countex image")
    subprocess.run(f"img2sixel {countexImg_p}", shell=True)
    #subprocess.run(f"img2sixel {att_p}", shell=True)
    
    """
    if os.path.exists(origImg_p):
        subprocess.run(["img2sixel", origImg_p])
    else:
        print(f"Couldn't find {origImg_p}")
    if os.path.exists(countexImg_p):
        subprocess.run(["img2sixel", contexImg_p])
    else:
        print(f"Couldn't find {countexImg_p}")
    """
def main():
    for i in range(1,23469): #4422から続きチェック
        id = i
        print(f"Showing the result of id: {id}.")
        origImg_path = (f"./CountEx_Images/'test-orig_qid_tensor([[{id}]]).png'")
        countexImg_path = (f"./CountEx_Images/'test_qid_tensor([[{id}]]).png'")
        att_path = (f"./Attention_Maps/'att_qid_tensor([[{id}]]).png'")
        question = (f"./Questions_and_Answers/'question_qid_tensor([[{id}]]).txt'")
        answer1 = (f"./Questions_and_Answers/'ans1_qid_tensor([[{id}]]).txt'")
        answer2 = (f"./Questions_and_Answers/'ans2_qid_tensor([[{id}]]).txt'")
    
        showImg(origImg_path, countexImg_path, att_path, question, answer1, answer2)
        
        i += 1
        #time.sleep(5)
        time.sleep(0.8)

if __name__ == "__main__":
    main()
