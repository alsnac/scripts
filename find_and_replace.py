import sys

# creating a variable and storing the text
# that we want to search
filename = sys.argv[1]

# creating a variable and storing the text
# that we want to search
search_text = ["Dataset/Dataset-25-Nov-2021/Pragas/anticarsia_gemmatalis/","Dataset/Dataset-25-Nov-2021/Pragas/dichelops_melacanthus/"]
  
# creating a variable and storing the text
# that we want to add
replace_text = ''
  
# Opening our text file in read only
# mode using the open() function
with open(filename, 'r') as file:
  
    # Reading the content of the file
    # using the read() function and storing
    # them in a new variable
    data = file.read()
  
    # Searching and replacing the text
    # using the replace() function
    for s_text in search_text:
      data = data.replace(s_text, replace_text)
  
# Opening our text file in write only
# mode to write the replaced content
with open(filename, 'w') as file:
  
    # Writing the replaced data in our
    # text file
    file.write(data)
  
# Printing Text replaced
print("Replaced:"+filename)