import re

# Read the input text from the file
#input_file_path = 'TSP100/alpha0beta100v3'
#input_file_path = 'TSP1000/alpha0beta100T004v3'

#input_file_path = 'TSP100/Rstartalpha0beta100v3'
input_file_path = 'TSP1000/Rstartalpha0beta50v3'
input_file_path = 'TSP1000/allzerosv1'
input_file_path = 'TSP1000/searchA0v1'
#input_file_path = 'TSP1000/allzeros008v1'
#input_file_path = 'TSP200/Nsearchv7'
#input_file_path = 'TSP100/smalltime'
#input_file_path = '/home/fs01/ym499/TSP200/Nsearchv1'
#input_file_path = 'TSP100/Nsearchv2'
input_file_path = 'TSP500/Nsearchv10'
#input_file_path = 'TSP1000/longsearchv1'
#input_file_path = 'TSP1000/longsearchzerov1'
#input_file_path = '/mnt/beegfs/bulk/mirror/ym499/UTSP/Search/results3'
#input_file_path = 'TSP200/Nshortv3'
#input_file_path = '/mnt/beegfs/bulk/mirror/ym499/UTSP/Search/Nresults'
#input_file_path = 'TSP1000/Shortsearchv5'
input_file_path = 'Nsearchv1'
avg_gap_floats = []

# Open the file and extract the float values
with open(input_file_path, 'r') as file:
    file_content = file.read()

    # Use regular expression to find the float values after "Avg_Gap:"
    avg_gap_values = re.findall(r"Avg_Gap: ([-+]?\d*\.\d+|\d+)", file_content)

    # Convert the extracted strings to float values
    avg_gap_floats = [float(value) for value in avg_gap_values]

# Print the extracted float values
count = 0
sum_of_Value = 0.0
for value in avg_gap_floats:
    count = count + 1
    sum_of_Value += value
    print(value)

print('Average Gap:%.8f'%(sum_of_Value/count))
