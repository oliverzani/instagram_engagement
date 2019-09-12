
'''
This script will collect a predefined number of instagram posts from a given instagram account.

Inputs:
    
    1. Replace the value of 'num_images' on line 25 with the amount of posts you want to collect. (Default 100)
        
    2. Replace the value of 'account_name' on line 26 with the instagram account you want to collect from.
           Note: The account_name must be identical to how it appears on instagram itself
                 (Also for obvious reasons the account must be public)
              
    3. Replace the value of 'output_file' on line 27 with the name of the desired output file
          
Outputs:

    1. The Captions, Likes, Comments, and image URLs will be written to the output file you named on line 27 
'''

#Necessary Packages
import instaloader
import  pandas as pd

#Set Parameters here
num_images = 800
account_name = 'marke_miller' 
output_file = 'marke_miller.xlsx'

#Create an Instaloader to collect images from the input account
loader = instaloader.Instaloader()
df=pd.DataFrame()
posts = instaloader.Profile.from_username(loader.context, account_name).get_posts()

cur=0
for post in posts:
    df = df.append({'Caption': post.caption, 'Likes': post.likes, 'Comments': post.comments, 'Date':post.date, 'URL': post.url  }, ignore_index=True)
    cur = cur+1
    if cur>num_images:
        break
    
#Creating the Engagement Score and outputting to an excel file
max_likes = max(df['Likes'])
max_comments = max(df['Comments'])

df['Likes_Normalized'] = df['Likes']/max_likes
df['Comments_Normalized'] = df['Comments']/max_comments

df.to_excel(output_file,index=False)
print("Written to " + output_file)

