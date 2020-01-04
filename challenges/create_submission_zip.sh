#!/bin/bash
# File              : create_submission_zip.sh
# Author            : Yanqing Wu <meet.yanqing.wu@gmail.com>
# Date              : 02.01.2020
# Last Modified Date: 04.01.2020
# Last Modified By  : Yanqing Wu <meet.yanqing.wu@gmail.com>

rm ./Yanqing_Wu_Challenge_Report.pdf
rm ./Yanqing_Wu.zip
cp ./reports/main.pdf Yanqing_Wu_Challenge_Report.pdf
zip Yanqing_Wu.zip ./Yanqing_Wu_Challenge_Report.pdf ./Yanqing_Wu_Code.py
