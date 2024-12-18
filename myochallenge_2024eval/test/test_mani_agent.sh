#!/bin/bash
# This script test the communication of the agent with the environment
GreenBK='\033[1;42m'
RedBK='\033[1;41m'
RC='\033[0m'

export PYTHONPATH="./utils/:$PYTHONPATH"
export PYTHONPATH="./agent/:$PYTHONPATH"
export PYTHONPATH="./environment/:$PYTHONPATH"

python environment/test_mani_environment.py &

# TO BE REPLACED WITH A DOCKER --> docker run myochallengeeval_mani_agent

python agent/agent_maniMPL.py
# docker run myochallengeeval_mani_agent

if [ $? -eq 0 ]; then
    printf "${GreenBK}Manipulation Agent script correctly connecting with the environment!${RC} \n"
else
    printf "${RedBK}Something is wrong! Check agent script!${RC} \n"
    echo FAIL
fi
