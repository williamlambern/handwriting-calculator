from NeuralNetwork import *
from tkinter import *



class Calculator:

    def __init__(self, master):

        self.points = []

        self.i = 0

        self.current = '' #This is where the current input is cached
        
        # NEURAL
        self.network = NeuralNetwork(400,25,12)
               
        # NETWORK
            
        self.master = master
        self.master.title('Calculator')
        self.frame = Canvas(self.master, width=500, height=100)
        self.text = self.frame.create_text(400,20, text='...',font="Aria 25", fill="red")
        self.frame.bind("<B1-Motion>", self.draw)
        self.frame.bind("<Button-2>", self.clear)
        self.frame.pack()

        

    def draw(self, event):
        #Draw a small oval because Tkinter is annoying
        if self.i == 1:
            self.frame.delete('all')
            self.points = []
            self.text = self.frame.create_text(400,20, text='...',font="Aria 25", fill="red")
            self.i = 0

        if len(self.points) > 0:
            if (event.x**2) - (self.points[len(self.points)-1][0]**2) + ((100-event.y)**2) - (self.points[len(self.points)-1][1]**2) > 10000:
                self.sendData(event)
                self.points = []            


        self.points.append([event.x , 100-event.y]) # y is messed up?
             
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        self.frame.create_oval( x1, y1, x2, y2, fill = "#476042" )

    def clear(self, event):
        self.sendData(event)
        self.i = 1
        self.frame.itemconfig(self.text, text=(self.current + ' = ' + str(self.calculate(self.current))))
        self.current = ''
        

    def calculate(self, equation):
        try:
            return eval(equation)
        except:
            return '?'

    def sendData(self, event):
        toSend = []
        formattedData = [[0.0 for i in range(20)] for i in range(20)]

        furthestLeft = 9999
        furthestRight = -9999
        furthestUp = -9999
        furthestDown = 99999

        for i in range(len(self.points)):          
            if self.points[i][0] < furthestLeft:
                furthestLeft = self.points[i][0] 
            if self.points[i][0] > furthestRight:
                furthestRight = self.points[i][0] 
            if self.points[i][1] < furthestDown:
                furthestDown = self.points[i][1] 
            if self.points[i][1] > furthestUp:
                furthestUp = self.points[i][1] 

        # edit all x,y based on the bottom left point (like a graph axis)

        for i in range(len(self.points)):
            self.points[i][0] -= furthestLeft
            self.points[i][1] -= furthestDown
            

        # find the factor to make the x , y fit into a 20x20 grid

        toChangeX = 20/(furthestRight+1 - furthestLeft)
        toChangeY = 20/(furthestUp+1 - furthestDown)

        for i in range(len(self.points)):
            
            self.points[i][0] = int(self.points[i][0] * toChangeX)
            self.points[i][1] = int(self.points[i][1] * toChangeY) #int as each pixel is represented by a whole number. Trunc so doesnt = 20

        # create an array 400 long with each 'pixel' represented e.g [(0,0) ... (0,19) , (1,0) ..... (19,19)]

        for i in range(len(self.points)):         
            try:
                formattedData[18-self.points[i][1]][self.points[i][0]] = 0.5
            except:
                pass
            try:
                formattedData[20-self.points[i][1]][self.points[i][0]] = 0.5
            except:
                pass
            try:
                formattedData[19-self.points[i][1]][self.points[i][0]+1] = 0.5
            except:
                pass
            try:
                formattedData[18-self.points[i][1]][self.points[i][0]+1] = 0.5
            except:
                pass
            try:
                formattedData[20-self.points[i][1]][self.points[i][0]+1] = 0.5
            except:
                pass
            try:
                formattedData[18-self.points[i][1]][self.points[i][0]-1] = 0.5
            except:
                pass
            try:
                formattedData[20-self.points[i][1]][self.points[i][0]-1] = 0.5
            except:
                pass

        for i in range(len(self.points)):
            formattedData[19-self.points[i][1]][self.points[i][0]] = 1.0  # y then x because of 2D array
            
        
        for a in range(20):
            for b in range(20):
                toSend.append(formattedData[a][b])
        '''
        for i in range(20):    
            print(formattedData[i])
        '''
        
        results = self.network.feedforward(np.array(toSend))
        best_guess = ''
        best_score = -1
        for i in range(len(results)):
            if results[i] > best_score:
                best_score = results[i]
                if i == 10:
                    best_guess = '+'
                if i == 11:
                    best_guess = '-'
                if i < 10:
                    best_guess = i
        
        self.current += str(best_guess)
        self.frame.itemconfig(self.text, text=self.current)

        
        


