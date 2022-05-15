# CS-456 Artificial Neural Networks, Mini-project 1
- [ ] Aggiungere requirements.txt per sns e altri pacchetti aggiunti
- [x] Trovare modo per aggiungere legenda dopo a tutti i plot (altrimenti bisogna salvare a mano tutto)
- [ ] Rivedere le figsize
- [ ] Rileggere/completare introduzione nel report

## QUESTIONS
### 2. Q-Learning
- [x] Measure performance
  - [x] Implementazione
  - [x] Documentazione (Ivan)

- [x] **Question 1**
  - [x] Implementare Q-Learning algo
    - [x] Implementazione
    - [x] Documentazione (Ivan) 
  - [x] Funzione per plot
    - [x] Implementazione
    - [x] Documentazione (Fede) 
  - [x] Test con epsilon a piacere (eps = 0.1)
  - [x] Commento plot (<50 parole) (Ivan) (Cresce velocemente all'inizio e poi satura)
  - [x] Review della risposta nel report (Fede)


- [x] **Question 2**
  - [x] Implementare decreasing epsilon
    - [x] Implementazione
    - [x] Documentazione (Ivan)
  - [x] Testare effetto di n*
  - [x] Miglioramento performance? Effetto di n*? (<200 parole))
  - [x] Con n* più alto all'inizio si gioca più random e la reward è minore (Fede)
  - [x] Review della risposta nel report (Ivan)
  

- [ ] **Question 3**
  - [x] Funzione per plot performance
    - [x] Implementazione
    - [x] Documentazione della parte aggiuntiva (Fede)
    - [x] Legenda unica per M_opt e M_rand
    - [x] Da discutere se va bene con la linea verticale per limite asintotico di epsilon
  - [x] Misurare performance
  - [x] Plot performance e descrizione (<100 parole)
  - [x] n* grandi rendono M_opt instabile e M_rand cresce più lentamente (Fede)
  - [ ] Review della risposta nel report (Ivan) ---> Riletto, ma bisogna parlarne

- [ ] **Question 4** (Ivan)
  - [ ] Trovare il miglior n* (attorno a 19000, da giustificare).
  - [x] Testare diversi epislon_opt
  - [x] Plot performance per diversi epsilon
  - [x] Commento (<250 parole)
  - [x] Giustificare scelta e_opt e perchè gli altri no. Con e_opt = 0 non vedo gli stati in cui potrei vincere. Con e_opt = 1 non imparo a difendermi dalla policy 
  ottima e provo a vincere invece di coprire l'avversario, contando sul fatto che lui gioca a caso.
  - [ ] Tagliare numero parole
  - [x] Review della risposta nel report (Fede)
  
- [ ] **Question 5**
  - [ ] Migliori M_opt, M_rand 
  - Da fare alla fine con seed fissato

- [ ] **Question 6** (Ivan)
  - [x] Domanda teorica (< 150 parole)
  - [ ] Tagliare numero parole
  - [x] Review della domanda nel report (Fede)
  - [ ] Chiedere per inizializzazione dei Q-values


- [ ] **Question 7**
  - [x] Q-learning by self-practice
    - [x] Implementazione
    - [x] Documentazione (Fede)
  - [x] Learning-by-self-practice con diversi epsilon
  - [x] M_opt e M_rand per diversi valori di epsilon
  - [x] Riesce ad imparare? Effetto di epislon? Commento (<100 parole) (Fede)
  - [ ] Review della risposta nel report (Ivan) ---> Riletto, ma bisogna parlarne


- [ ] **Question 8** (Fede)
  - [x] Usare epsilon(n)
  - [x] Plot performance per diversi n*.
  - [x] Aiuta rispetto a epsilon fisso? Effetto di n*?  Commento (<100 parole)
  - [x] Review della risposta nel report (Ivan)

 
- [ ] **Question 9**
  - [x] Usare epsilon(n)
  - [ ] Migliore performance?
  - [ ] Fare alla fine con seed fissato.


- [ ] **Question 10**
  - [x] Usare epsilon(n)
  - [x] Funzione per visualizzare Q-values su stato
    - [x] Implementazione (Fede)
      - [x] Scala uniforme
      - [x] Croci e cerchi  
    - [x] Documentazione (Fede)
  - [x] Scegliere configurazioni significative
  - [x] Unire le tre immagini in una sola (non si possono fare subplots con ggplot su Python)
  - [x] Commento (<200 parole) (Win, block win e fork) (Ivan)
  - [ ] Da tagliare il numero di parole se la caption conta nelle parole e da aggiungere un altro commentino se la caption non conta.
  - [x] Allineare le subfigures a [t] e non [b] 
  - [x] Review della risposta nel report (Fede) ---> Fatto, ci sono solamente alcuni dettagli
  

### 3. Deep Q-Learning
- [x] Tutorial Keras
- [x] Impostare la Neural Network


- [ ] Optional (da fare)
  - [ ] Fine-tune learning rate
  - [ ] Try other optimizations (<300 parole, alla fine)


- [ ] **Question 11**
  - [x] Implementare DQN
    - [x] Implementation
    - [ ] Documentation (Fede)
  - [ ] Plot loss
  - [ ] Commento (<50 parole a figura)


- [ ] **Question 12**
  - [ ] Implementazione
  - [ ] Commento (<50 parole a figura)

  
- [ ] **Question 13**
  - [ ] Provare diversi n* (logspace/linspace?)
  - [ ] Plot performance (metrics) per diversi n*
  - [ ] Commento (<250 parole)
  - [ ] Settare l'hyperparameter n*


- [ ] **Question 14**
  - [ ] Comparare DQN con epsilon-optimal
  - [ ] Plot performance
  - [ ] Commento (<250 parole)


- [ ] **Question 15**
  - [ ] Migliore performance


- [ ] **Question 16**
  - [ ] Implementare learning by self practice con DQN
    - [ ] Implementazione
    - [ ] Documentazione
  - [ ] Plot performance al variare di epsilon fisso
  - [ ] Commento (<100 parole)


- [ ] **Question 17**
  - [ ] Provare diversi n* (logspace/linspace?)
  - [ ] Plot performance (metrics) per diversi n*
  - [ ] Commento (<250 parole)
  - [ ] Settare l'hyperparameter n*


- [ ] **Question 18**
  - [ ] Migliore performance


- [ ] **Question 19**
 - [ ] Usare epsilon(n)
  - [ ] Funzione per visualizzare Q-values su stato (riadattare la vecchia?)
    - [ ] Implementazione
    - [ ] Documentazione
  - [ ] Scegliere configurazioni significative
  - [ ] Commento (<200 parole)

### 4. Comparing Q-Learning with Deep Q-Learning
- [ ] **Question 20**
  - [ ] Tabella con: 
    - [ ] migliori performance
    - [ ] training time (da fare su un unico computer)


- [ ] **Question 21**
  - [ ] Comparison (<300 parole)


## REPORT
- [x] Template
- [x] Fare sync con Overleaf per immagini 

## Authors
- Federico Betti
- Ivan Bioli
