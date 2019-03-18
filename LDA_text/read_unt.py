import json

def read_line(line, tweets):
    tweet_dict = json.loads(line)
    tweets.append(tweet_dict["text"])

file_name = "/corral-repl/utexas/Trump-Tweets/Harvey/unt_data/harvey_twitter_dataset/02_archive_only/HurricaneHarvey.json"
count = 0.0
tweets = []
num_lines = sum(1 for line in open(file_name))
storm = 'unt'

with open(file_name) as f:
    for line in f:
        read_line(line, tweets)
        count += 1
        print('PROGRESS: ({c}%)'.format(c=(int(count / num_lines * 100))))

print(len(tweets))
with open("storm_extracts/{storm}".format(storm=storm), "wb") as fp:   #Pickling
        pickle.dump(tweets, fp)



# chunked file reading
# from __future__ import division
# import os

# def get_chunks(file_size):
#     chunk_start = 0
#     chunk_size = 0x20000  # 131072 bytes, default max ssl buffer size
#     while chunk_start + chunk_size < file_size:
#         yield(chunk_start, chunk_size)
#         chunk_start += chunk_size

#     final_chunk_size = file_size - chunk_start
#     yield(chunk_start, final_chunk_size)

# def read_file_chunked(file_path):
#     with open(file_path) as file_:
#         file_size = os.path.getsize(file_path)

#         print('File size: {}'.format(file_size))

#         progress = 0

#         for chunk_start, chunk_size in get_chunks(file_size):

#             file_chunk = file_.read(chunk_size)

#             # do something with the chunk, encrypt it, write to another file...
#             print("CHUNK")
#             print(file_chunk)

#             progress += len(file_chunk)
#             print('{0} of {1} bytes read ({2}%)'.format(
#                 progress, file_size, int(progress / file_size * 100))
#             )

# if __name__ == '__main__':
#     read_file_chunked('some-file.gif')

# import multiprocessing as mp,os

# def process_wrapper(chunkStart, chunkSize):
#     file_name = "/corral-repl/utexas/Trump-Tweets/Harvey/unt_data/harvey_twitter_dataset/02_archive_only/HurricaneHarvey.json"
#     with open(file_name) as f:
#         f.seek(chunkStart)
#         lines = f.read(chunkSize).splitlines()
#         for line in lines:
#             print(line)

# def chunkify(fname,size=1024*1024):
#     fileEnd = os.path.getsize(fname)
#     with open(fname,'r') as f:
#         chunkEnd = f.tell()
#         while True:
#             chunkStart = chunkEnd
#             f.seek(size,1)
#             f.readline()
#             chunkEnd = f.tell()
#             yield chunkStart, chunkEnd - chunkStart
#             if chunkEnd > fileEnd:
#                 break

# #init objects
# file_name = "/corral-repl/utexas/Trump-Tweets/Harvey/unt_data/harvey_twitter_dataset/02_archive_only/HurricaneHarvey.json"
# cores = 24
# pool = mp.Pool(cores)
# jobs = []

# #create jobs
# for chunkStart,chunkSize in chunkify(file_name):
#     jobs.append( pool.apply_async(process_wrapper,(chunkStart,chunkSize)) )

# #wait for all jobs to finish
# for job in jobs:
#     job.get()

# #clean up
# pool.close()