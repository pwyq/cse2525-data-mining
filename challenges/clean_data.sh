#!/bin/bash
# File              : clean_data.sh
# Author            : Yanqing Wu <meet.yanqing.wu@gmail.com>
# Date              : 27.12.2019
# Last Modified Date: 27.12.2019
# Last Modified By  : Yanqing Wu <meet.yanqing.wu@gmail.com>

cp ./data/ratings_comb.csv ./origin_data/ratings_comb.csv
rm -rf ./data
cp -r ./origin_data ./data
