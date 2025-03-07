

#Implemented as an array of chars
class My_string():
    #Strings can be initialized in a couple ways such as
    # with nothing or with a string already created
    def __init__(self) -> None: # <-These functions should not return anything
        self.string = []
    
    def __init__(self, string = str) -> None:
        self.string = []
        if type(string) != str:
            raise TypeError()
        for char in string:
            self.string.append(char)
        

    #Being able to slice a string into pieces is very useful allowing you to take a look at portions of the
    # the string and to perform different operations depending on what is revealed from those splits    
    # The intention to put end first is intentional much like [:] in python this is meant to be similar
    # if you only put the end that is fine and it will give you
    def split(self, end, start = 0) -> str:

        if start < 0:
            raise IndexError
        if end > self.length():
            raise IndexError

        build_str = ""
        if start > end: # This is if they are asking for a str slice in order
            for x in range(end-start):
                build_str += self.string[x + start]
        else:

            for x in range(start - end): # This is if they are asking fora  str slice in reverse
                build_str += self.string[end - x]

        
        return build_str

    # Alot of the time we need to find the length of a string
    # Instead of using additional memory to keep track we can make
    # a function that can find the answer on the fly. This function
    # is O(n) but if we had a value to keep track of the length it would be
    # O(1)   
    def length(self) -> int:
        count = 0
        for char in self.string:
            count += 1
        
        return count

    # we want to be able to print our string out especially since we are implementing it
    # as an array/list
    # This will print each character in the list one at a time and will put an end line after it
    # is done
    def print(self) -> None:
        for char in self.string:
            print(char, end = "")
        print()


    # Adding another string onto this current string
    # because we are implementing as an array we can just append the other string onto ours
    # Normally in other languages we will have to worry about size of the array and indexing
    # out of bounds but in python we do not
    def add_string(self, other_string = 'My_string') -> None:
        for char in other_string.string:
            self.string.append(char)



