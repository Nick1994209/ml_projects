#!/usr/bin/env bash


FILE_PATH="review_polarity.tar.gz"

echo $FILE_PATH
wget --output-document=$FILE_PATH http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

tar -xvzf $FILE_PATH
