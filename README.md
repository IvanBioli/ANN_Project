# CS-456 Artificial Neural Networks, Mini-project 1
This repository contains the implementation of Q-Learning and Deep Q-Learning algorithms for training an agent to play Tic-Tac-Toe.

## Setup

All instructions are relative to the root directory of the repository.

### Installation
1. Clone the following repository:
```python
git clone --recursive https://github.com/IvanBioli/ANN_Project
``` 
1. Detailed requirements for the needed packages are available in requirements.txt. To install the needed packages, please run:
```
pip install -r requirements.txt
```

## Repository Description
* `figures` - description
* `report` - description
* `results` - description
* `code` - description
  * `file1.py` - description
  * `file2.ipynb` - description

## Usage
### Reproducing the results in the report
We provide a unique notebook containing the answers to all the questions of the project (except for Question 6). The notebook should be used as follows: concerning the plots presented in the report (see below), setting in the very first cells
```python
Train = False
Load = not Train
```
will load direcly our results from the provided dictionaries in the folder results and will therefore show for most questions both the plot presented in the report (see below) and a second plot showing all the experimented values. On the other hand, since our results are obtained as an average over 10 training runs for Q-Learning and 4 runs for Deep Q-Learning, if one wants to perform training simply set
```python
Train = True
Load = not Train
```
and in the following cell, we recommend for computational time reasons setting
```python
num_avg = 1
```

### Example of usage for the Q-Learning and DQN algorithms
- Q-Learning training
```python
env = TictactoeEnv()  # set the environment
alpha = 0.05  # learning rate
gamma = 0.99  # discount factor
epsilon_exploration = 0.1  # exploration rate
test_freq = 250  # testing frequency during training
against_opt = True  # or self_practice = True, depending on the desired training method (note that one of the two must be set, otherwise ValueError is raised)
Q, stats = q_learning(env, alpha=alpha, gamma=gamma, epsilon_exploration=epsilon_exploration, test_freq=test_freq, against_opt=against_opt)  # return Q-values and training stats
```

- Deep Q-Learning training
```python
env = TictactoeEnv()  # set the environment
alpha = 1e-4  # learning rate
gamma = 0.99  # discount factor
epsilon_exploration = 0.1  # exploration rate
test_freq = 250  # testing frequency during training
against_opt = True  # or self_practice = True, depending on the desired training method (note that one of the two must be set, otherwise ValueError is raised)
model, stats = deep_q_learning(env, alpha=alpha, gamma=gamma, epsilon_exploration=epsilon_exploration, test_freq=test_freq, against_opt=against_opt)  # return model network and training stats
```

For the questions in which the optimal values of the performance measures, we both show a sample run for the optimal values of the parameter and then we load again the results from dictionaries where the results presented in the report are stored.

## Report 

## Remarks

## Authors
- Federico Betti
- Ivan Bioli

## REMAINING TO DO
- [ ] Aggiungere requirements.txt per sns e altri pacchetti aggiunti, preparare bene ReadMe (Fede)
- [x] Trovare modo per aggiungere legenda dopo a tutti i plot (altrimenti bisogna salvare a mano tutto)
- [ ] Rivedere le figsize (Ivan)
- [ ] Rileggere/completare introduzione nel report
- [x] Merge finale dei due notebook per riproducibilità
- [x] Tenere solamente dizionari finali e ultima run del notebook finale per controllare tutto (Ivan)
- [x] Documentazione di tutto il codice, rivedere (Fede)
- [x] Spiegare ulteriormente oscillazioni di M_opt
- [ ] Preparare file separato con conteggio parole (Fede)
- [x] Legenda unica per plot rewards e loss?
- [ ] Come facciamo per mostrare che va meglio alpha = 0.25? (Ivan: NON LO FACCIAMO VEDERE)
- [ ] Teniamo train_avg con dqn parametro o deep_train_avg? Funzionano entrambe (Ivan: unica funzione con parametro meglio)

## QUESTIONS
### 2. Q-Learning

- [x] **Question 1**

- [x] **Question 2** (Rileggere)

- [x] **Question 3** (Rileggere)

- [x] **Question 4**
  
- [x] **Question 5**

- [x] **Question 6**

- [x] **Question 7** 

- [x] **Question 8** (Rileggere)
 
- [x] **Question 9**

- [x] **Question 10** (Rileggere)

### 3. Deep Q-Learning

- [x] **Question 11**

- [ ] **Question 12** (Insieme)
  
- [x] **Question 13** (Rileggere, entrambi)

- [x] **Question 14** (Rileggere)

- [x] **Question 15**

- [x] **Question 16** (Rileggere, entrambi)

- [x] **Question 17** (Rileggere, entrambi)

- [x] **Question 18**

- [x] **Question 19** (Rileggere)

### 4. Comparing Q-Learning with Deep Q-Learning
- [ ] **Question 20** (Fede)
  - [ ] Tabella con: 
    - [ ] migliori performance
    - [ ] training time 
    - [ ] Implementare training time computation

- [ ] **Question 21** (Entrambi scrivono una brutta su Overleaf)
  - [ ] Comparison (<300 parole)
