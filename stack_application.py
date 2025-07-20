class Browser:
    ''' A broswer class where attribute of a class is list, which contain all data browsing history'''
    
    def __init__(self):
        self.browsing_backStack = []
        self.browsing_forwardStack = []
    
    # method to append browsing in your list and  
    def start_browsing (self,url):
        self.browsing_backStack.append(url)
      
    
    # browsing history based on LIFO
    def browsing_history(self):
        if not self.browsing_backStack:
            print('Nothing to go backward')
        else:
            for url in reversed(self.browsing_backStack):
                print(url)
                
    # method for back button where actual stack pricipal LIFO will applied
    def back_button(self):
        if len(self.browsing_backStack)>1:
            last_visitied_site =  self.browsing_backStack.pop()
            self.browsing_forwardStack.append(last_visitied_site)
            print(f'Going backward to :{self.browsing_backStack[1:]}')
        else:
            print('Nothing to go back')
    
    def forward_button(self):
        if len(self.browsing_forwardStack)>1:
            url = self.browsing_forwardStack.pop()
            self.browsing_backStack.append(url)
            print(f'forward to : {url}')
        else:
            print('you just started browsing!')

 __init__ == __main()__:
  browsing = Browser()
  browsing.start_browsing('abc.com')
  browsing.start_browsing('xyz.com')
  browsing.start_browsing('tuv.com')
  print('\n your Browsing history')
  browsing.browsing_history()
  browsing.back_button()
  #browsing.browsing_history()
  browsing.start_browsing('zte.com')
  browsing.browsing_history()
  browsing.back_button()
  browsing.forward_button()

