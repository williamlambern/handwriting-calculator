from Calculator import *


def main():
    root = Tk()
    cal = Calculator(root)
    c = input('Do you want to train the network (y/n) ')
    if c== 'y':
       cal.network.train()
    root.mainloop()

if __name__ == '__main__':
    main()
