import random
import string 

class Service:
    
    def generate_folder_name(self):
        length = 15
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str