# mkdir data
mkdir data

# MR: Movie Review Data
wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
mv rt-polaritydata.tar.gz ./data/
cd ./data
tar -xzf rt-polaritydata.tar.gz

# Pretrained word embeddings
FILE_ID=0B7XkCwpI5KDYNlNUTTlSS21pQmM
FILE_NAME=GoogleNews-vectors-negative300.bin.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
gzip -d GoogleNews-vectors-negative300.bin.gz
