import tkinter as tk


base=tk.Tk()  #this is the main frame
root=tk.Frame(base)  #Really this is not necessary -- the other widgets could be attached to "base", but I've added it to demonstrate putting a frame in a frame.
root.grid(row=0,column=0)
scoreboard=tk.Frame(root)
scoreboard.grid(row=0,column=0,columnspan=2)

###
#Code to add stuff to scoreboard ...
# e.g.
###
scorestuff=tk.Label(scoreboard,text="Here is the scoreboard")
scorestuff.grid(row=0,column=0)
#End scoreboard

#Start cards.
cards=tk.Frame(root)
cards.grid(row=1,column=0)
###
# Code to add pitcher and batter cards
###
clabel=tk.Label(cards,text="Stuff to add cards here")
clabel.grid(row=0,column=0)
#end cards

#Offense/Defense frames....
offense=tk.Frame(root)
offense.grid(row=1,column=1)
offense.isgridded=True #Dynamically add "isgridded" attribute.
offense_label=tk.Label(offense,text="Offense is coolest")
offense_label.grid(row=0,column=0)

defense=tk.Frame(root)
defense.isgridded=False
defense_label=tk.Label(defense,text="Defense is coolest")
defense_label.grid(row=0,column=0)

def switchOffenseDefense():
    print("Called")
    if(offense.isgridded):
        offense.isgridded=False
        offense.grid_forget()
        defense.isgridded=True
        defense.grid(row=1,column=1)
    else:
        defense.isgridded=False
        defense.grid_forget()
        offense.isgridded=True
        offense.grid(row=1,column=1)


switch_button=tk.Button(root,text="Switch",command=switchOffenseDefense)
switch_button.grid(row=2,column=1)

root.mainloop()





import tkinter as tk
import tkinter.messagebox as messagebox

board = [ [None]*10 for _ in range(10) ]

counter = 0
root = tk.Tk()

def check_board():
    freespaces = 0
    redspaces = 0
    greenspaces = 0
    for i,row in enumerate(board):
        for j,column in enumerate(row):
            if board[i][j] == "red":
                redspaces += 1
            elif board[i][j] == "green":
                greenspaces += 1
            elif board[i][j] == None:
                freespaces += 1

    if freespaces == 0:
        if greenspaces > redspaces:
            winner = "green"
        elif greenspaces < redspaces:
            winner = "red"
        else:
            winner = "draw"

        if winner != "draw":
            messagebox.showinfo("Game Over!",winner+" wins!")
        else:
            messagebox.showinfo("Game Over!","The game was a draw!")




def on_click(i,j,event):
    global counter
    if counter < 100:
        if board[i][j] == None:
            color = "green" if counter%2 else "red"
            enemycolor = "red" if counter%2 else "green"
            event.widget.config(bg=color)
            board[i][j] = color
            for k in range(-1,2):
                for l in range(-1,2):
                    try:
                        if board[i+k][j+l] == enemycolor:
                            board[i+k][j+l] = color
                    except IndexError:
                        pass
            counter += 1
            global gameframe
            gameframe.destroy()
            redraw()
            root.wm_title(enemycolor+"'s turn")
        else:
            messagebox.showinfo("Alert","This square is already occupied!")
        check_board()


def redraw():
    global gameframe
    gameframe = tk.Frame(root)
    gameframe.pack()

    for i,row in enumerate(board):

        for j,column in enumerate(row):
            name = str(i)+str(j)
            L = tk.Label(gameframe,text='    ',bg= "grey" if board[i][j] == None else board[i][j])
            L.grid(row=i,column=j,padx='3',pady='3')
            L.bind('<Button-1>',lambda e,i=i,j=j:on_click(i,j,e))


redraw()
root.mainloop()







#######################################

#Imports all (*) classes,
#atributes, and methods of tkinter into the
#current workspace

from tkinter import Tk, StringVar, Label, Entry, Button, W, E

root = Tk()
root.title('how to get text from textbox')


#**********************************
mystring = StringVar()

####define the function that the signup button will do
def getvalue():
    print(mystring.get())
    return mystring.get()

#*************************************

Label(root, text="Text to get").grid(row=0, sticky=W)  #label
Entry(root, textvariable = mystring).grid(row=0, column=1, sticky=E) #entry textbox

WSignUp = Button(root, text="print text", command=getvalue).grid(row=3, column=0, sticky=W) #button


############################################
# executes the mainloop (that is, the event loop) method of the root
# object. The mainloop method is what keeps the root window visible.
# If you remove the line, the window created will disappear
# immediately as the script stops running. This will happen so fast
# that you will not even see the window appearing on your screen.
# Keeping the mainloop running also lets you keep the
# program running until you press the close buton
root.mainloop()
# while True :
#     root.update()


