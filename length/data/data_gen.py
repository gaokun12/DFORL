data = []

for i in range(10):
    data.append('succ('+ str(i) +',' + str(i+1)+ ').')


max_int = 10
int_elemetns = [10,9,8,7,6,5,4,3,2,1]
all_slides = []
index = 0
while index < len(int_elemetns):
    first_part = int_elemetns[0:index]
    second_part = int_elemetns[index:len(int_elemetns)]
    all_slides.append(second_part)
    index += 1

for i in all_slides:
    for j in all_slides:
        if j[0] == i[0]-1:
            new_i = str(i).replace(', ','>')
            new_j = str(j).replace(', ','>')
            data.append('cons('+new_i+','+new_j+').')
for i in all_slides:
    new_i = str(i).replace(', ','>')
    data.append('length('+new_i+','+str(len(i))+').')

        
file_path = 'NLP/length/data/'
with open(file_path + 'length.nl', 'w') as f:    
    for i in range(int(len(data))):
        print(data[i], file=f)
    f.close()


    
#     f.close()
