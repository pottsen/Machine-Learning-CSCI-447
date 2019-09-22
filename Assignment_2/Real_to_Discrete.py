

def handle_files():
    #files = [["filename" , column_of_class, [discrete_values_columns]], ......]
    files = [["abalone", 0, [1,2,3,4,5,6,7]],
             #["car", 6, []],
             ["forestfires", 12, [0,1,2,3,12]],
             #["machine", 0, [0,1]],
             ["segmentation", 0, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]],
             ["wine", 0, [1,2,3,4,5,6,7,8,9,10,11,12,13]]] 
    
    for i in range(len(files)):
        descretize(files[i])
        

if __name__ == "__main__":
    handle_files()