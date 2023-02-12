# pacman_adversarial_search_algorithms
Various search algorithms in Pacman for finding optimal moves when dealing with adversarial agents, mainly minimax (including alpha beta pruning) and expectimax

All the code has been provided by UC Berkeley, apart from the search algorithms and some evaluation functions.

## Reflex Agent
To play with the reflex agent, which uses A* Star Search in the evaluation function to pick the best action at the given moment, you can use the following commands:
```
python pacman.py -p ReflexAgent -l testClassic
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2
```

## Minimax
To play with the agent that uses the minimax algorithm, which makes a move based on the worst possible outcome from adversarial agents, you can use the following commands:
```
python autograder.py -q q2
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
python pacman.py -p MinimaxAgent -l mediumClassic -a depth=4
python pacman.py -p MinimaxAgent -l openClassic -a depth=4
```

## Minimax with Alpha-Beta Pruning
to play with the agent that uses the minimax algorithm, but with a more efficient time complexity since the decision tree has been pruned due to alpha and beta cut-off limits,
you can use the following commands:
```
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
python pacman.py -p AlphaBetaAgent -a depth=3 -l minimaxClassic
python pacman.py -p AlphaBetaAgent -l mediumClassic -a depth=4
python pacman.py -p AlphaBetaAgent -l openClassic -a depth=4
python autograder.py -q q3
```

## Expectimax
To play with the agent that uses the expectimax algorithm, which relies on probability of certain consequences occuring and thus introducing some randomness to moves,
you can use the following commands:
```
python autograder.py -q q4
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
python pacman.py -p ExpectimaxAgent -a depth=3 -l smallClassic
python pacman.py -p ExpectimaxAgent -l mediumClassic -a depth=4
python pacman.py -p ExpectimaxAgent -l openClassic -a depth=4
```
You can also compare the difference in outcomes with the minimax algorithm and expectimax algorithm in a futile situation like trappedClassic:
```
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
```
You should find that due to going off probability rather than looking at the worst outcome, expectimax can sometimes win while minimax never wins.

## Evaluation Function
To play with an evaluation function that I created for various states in the game to make decisions, you can use the following commands:
```
python autograder.py -q q5
```
