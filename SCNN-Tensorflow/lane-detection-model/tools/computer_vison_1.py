
from tkinter import *
import imageio
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import glob
import cv2
from advanced_lane import *
import os
import cv2
import numpy as np
import test_lanenet as t
from time import sleep

def create_video(path):
    img_array = []
    for filename in path:
        #video_name=filename[-59:-10]+".mp4"
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    #video_name=video_name.replace("\\","Z")
    #print(video_name)
    out = cv2.VideoWriter(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\Visual_output\Dl\video\video.mp4",cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    return out.release()

def cut_video(pathread,pathsave,file):
    vidcap=cv2.VideoCapture(pathread)
    success,image=vidcap.read()
    count=0
    #print(path)
    f=open(file,'w')
    while success:
        cv2.imwrite(pathsave+"//0000%d.jpg" % count,image)
        f.write(pathsave+"/0000%d.jpg" % count)
        f.write("\n")
        success,image=vidcap.read()
        count+=1
        #print("read a new frame: " , success)
def fileDialog1():
        filename = filedialog.askopenfilename( )
        #print(filename)
        video_name=filename[-19:-16]
        print("momo"+video_name)
        white_output = r'C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\Visual_output\cv\video\\'+video_name+'.mp4'
        clip1 = VideoFileClip(filename)
        init_lines(590)
        white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
        
def fileDialog():
        filename = filedialog.askopenfilename( filetype =
        (("jpg files","*.jpg"),("all files","*.*")) )
        #print(filename)
        for img_name in glob.glob(filename):
            img = cv2.imread(img_name)
            init_lines(img.shape[0])
            img2,L,R = pipeline1(img)
            image_name=img_name[-9:]
            img2=cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            plt.figure(figsize = (10,5))
            plt.imshow(img2)
            cv2.imwrite(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\Visual_output\cv\image\\"+image_name+".jpg",img2)
            parent_path = os.path.dirname(filename)[98:]
            parent_path = parent_path.replace('/', '\\')
            directory = os.path.join(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\experiments\predicts", 'vgg_SCNN_DULR_w9', parent_path)
            #print(directory)
            if not os.path.exists(directory):
                    os.makedirs(directory)
            file_exist = open(os.path.join(directory, os.path.basename(filename)[:-3] + 'exist.txt'), 'w')
            for cnt_img in range(1):
              cv2.imwrite(os.path.join(directory, os.path.basename(filename)[:-4] + '_' + str(cnt_img + 2) + '_avg.png'),L)
              cv2.imwrite(os.path.join(directory, os.path.basename(filename)[:-4] + '_' + str(cnt_img + 3) + '_avg.png'),R)
            file_exist.write('0 ')
            if ones_or_zeros_for_Right()==True:
                file_exist.write('1 ')
            else:
                file_exist.write('0 ')
            if ones_or_zeros_for_Left()==True:
                file_exist.write('1 ')
            else:
                file_exist.write('0 ')
            file_exist.write('0 ')
            file_exist.close()
                

def fileDialog2():
        l=Button(win1,text="dl",height=2,width=15,font=(25),bd=15,bg="#ff0000",fg="black",command=DL)
        file=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\data\CULane\video.txt"
        file2=open(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\data\CULane\list\test.txt","w")
        file3=open(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\demo_file\test_img.txt","w")
        filename = filedialog.askopenfilename( )
        path=filename[:-4]
        #file="../"+path[86:]
        try:
            os.makedirs(path)
            #print("Directory ",path," Created")
        except FileExistsError:
            pass
            print("Directory ",path," already exists")
        cut_video(filename,path,file)
        with open(file,'r') as f:
            for line in f:
              if line !="\n":
                #global v
                #v=line[-20:-16]
                image_name="../"+line[86:]
                file3.write(image_name)
                #print(image_name)
                image_name2=line[97:]
                file2.writelines(image_name2)
                #print(image_name2)
        sleep(5)        
        l.place(x=500,y=500)
        win1.pack( fill=BOTH, expand=True)

  
def DL():
  
    t.init()
    import main1 as m 
    #sleep(20)
    #from main1 import * 
    p=m.pat()
    path2=glob.glob(p+"\*.jpg")
    create_video(path2)
            
        
def fileDialog3():
        l=Button(win1,text="dl",height=2,width=15,font=(25),bd=15,bg="#ff0000",fg="black",command=DLL)
        filename = filedialog.askopenfilename( filetype =
        (("jpg files","*.jpg"),("all files","*.*")) )
        file=open(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\demo_file\test_img.txt","w")
        image_name="../"+filename[86:]
        image_name2=filename[97:]
        file2=open(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\data\CULane\list\test.txt","w")
        print(image_name2)
        print(image_name)
        file.write(image_name)
        file2.writelines(image_name2)
  
        sleep(5)
        #import test_lanenet as t
        l.place(x=500,y=500)
        win1.pack( fill=BOTH, expand=True)
        #t.init()


def DLL():
  
    t.init()
    import main1 as m 

    
        
def clearFrame(frame):
    # destroy all widgets from frame
    for widget in frame.winfo_children():
       widget.destroy()

    # this will clear frame and frame will be empty
    # if you want to hide the empty panel then
    frame.pack_forget()
def BACK():
    clearFrame(win1)
    root.title(" Lane line detection ")
    root.iconbitmap(r'C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\algorithm.ico')
    root.geometry("1200x700")
    root.configure(background="black")
    p3 = PhotoImage(file=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\111111.png")
    global win
    win=Frame(root)
    lm=Label(win,image=p3)
    lm.pack( fill=BOTH, expand=True)
    b1=Button(lm,text="Computer Vision",height=2,width=15,compound=CENTER,font=('TimesNewRoman', 14, 'bold'),bd=10,bg="SteelBlue1",fg="black",command=switch)
    b2=Button(lm,text="Deep Learning",height=2,width=15,compound=CENTER,font=('TimesNewRoman', 14, 'bold'),bd=10,bg="SteelBlue1",fg="black",command=switch2)
    b1.place(x=200,y=300)
    b2.place(x=800,y=300)
    b3=Button(lm,text="Exit",command=root.destroy,height=2,width=15,font=('TimesNewRoman', 14, 'bold'),bd=15,bg="SteelBlue4",fg="black")
    b3.place(x=475,y=500)
    win.pack(fill=BOTH, expand=True)
    root.resizable(0, 0)
    root.mainloop()
def switch():
    clearFrame(win)
    root.title("Lane line detection Using Computer Vision")
    root.iconbitmap(r'C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\Balance.ico')
    root.geometry("1200x700")

    global win1
    win1=Frame(root)
    llm=Label(win1,image=p2)
    

    
    b4=Button(llm,text="Image",height=2,width=15,compound=CENTER,font=(25),bd=20,bg="black",fg="ivory2",command=fileDialog)
    b7=Button(llm,text="Video",height=2,width=15,compound=CENTER,font=(25),bd=20,bg="black",fg="ivory2",command=fileDialog1)
    b4.place(x=500,y=200)
    b7.place(x=800,y=200)
    b5=Button(llm,text="Exit",command=root.destroy,height=2,width=15,font=(25),bd=15,bg="#ff0000",fg="ivory2")
    b5.place(x=1000,y=600)
    b6=Button(llm,text="Back",command=BACK,height=2,width=15,font=(25),bd=15,bg="blue",fg="ivory2")
    b6.place(x=0,y=0)
    llm.pack( fill=BOTH, expand=True)
    win1.pack(fill=BOTH, expand=True)
def switch2():
    clearFrame(win)
    root.title(" Lane line detection Using Deep learning ")
    root.iconbitmap(r'C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\A cup of coffee.ico')
    root.geometry("1200x700")    
    
    global win1
    win1=Frame(root)
    global llm
    llm=Label(win1,image=p1)
    
   
    b4=Button(llm,text="Image",height=2,width=15,compound=CENTER,font=(25),bd=10,bg="white",fg="black",command=fileDialog3)
    b1=Button(llm,text="Video",height=2,width=15,compound=CENTER,font=(25),bd=10,bg="white",fg="black",command=fileDialog2)
    b4.place(x=500,y=200)
    b1.place(x=700,y=200)
    b5=Button(llm,text="Exit",command=root.destroy,height=2,width=15,font=(25),bd=15,bg="#ff0000",fg="black")
    b5.place(x=1000,y=600)
    b6=Button(llm,text="Back",command=BACK,height=2,width=15,font=(25),bd=15,bg="blue",fg="ivory2")
    b6.place(x=0,y=0)
    llm.pack( fill=BOTH, expand=True)
    win1.pack(fill=BOTH, expand=True)

global root
root = Tk()
root.title(" Lane line detection ")
root.iconbitmap(r'C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\algorithm.ico')
root.geometry("1200x700")
root.configure(background="black")


p3 = PhotoImage(file=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\111111.png")
global win
win=Frame(root)
lm=Label(win,image=p3)
lm.pack( fill=BOTH, expand=True)


p2 = PhotoImage(file=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\1212.png")
p1 = PhotoImage(file=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\GUI\3333.png")
b1=Button(lm,text="Computer Vision",height=2,width=15,compound=CENTER,font=('TimesNewRoman', 14, 'bold'),bd=10,bg="SteelBlue1",fg="black",command=switch)
b2=Button(lm,text="Deep Learning",height=2,width=15,compound=CENTER,font=('TimesNewRoman', 14, 'bold'),bd=10,bg="SteelBlue1",fg="black",command=switch2)
b1.place(x=200,y=300)
b2.place(x=800,y=300)
b3=Button(lm,text="Exit",command=root.destroy,height=2,width=15,font=('TimesNewRoman', 14, 'bold'),bd=15,bg="SteelBlue4",fg="black")
b3.place(x=475,y=500)
win.pack(fill=BOTH, expand=True)
root.resizable(0, 0)
root.mainloop()

      
