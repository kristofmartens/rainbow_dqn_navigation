# DQN Navigation

As experienced data engineer I am not a big fan of using notebooks as deliverables for data scientists. It does not
encourage testing and makes it very hard to make the code production ready. Instead of encouraging data scientists to 
create notebooks, you should encourage them to create structured code that is easily tested/maintainable. For that 
reason I will not copy/paste the following code in a notebook

## Code structure
- banana: gym environment, allowing to plug in any environment. Add Banana.app folder in the banana/data directory
- model: Contains the neural network architecture used for this project
- rainbow_dqn_agent: Contains the code for the reinforcement learning agent. Not all rainbow dqs aspects are implemented
yet. This will be updated at a later time
- result: contains the model after training
- main.py: Contains the main program loop and a description of all parameter options, sensible defaults are there

## Training
```bash 
python3 src/main.py
```

## Evalutation
```bash 
python3 src/main.py --evaluation 
python3 src/main.py --evaluation --realtime
```

This gives me the following result:
```text
Episode: 1	Step: 300	Score: 18.0	Avg Score: 18.0
Episode: 2	Step: 600	Score: 14.0	Avg Score: 16.0
Episode: 3	Step: 900	Score: 13.0	Avg Score: 15.0
Episode: 4	Step: 1200	Score: 17.0	Avg Score: 15.5
Episode: 5	Step: 1500	Score: 17.0	Avg Score: 15.8
Episode: 6	Step: 1800	Score: 15.0	Avg Score: 15.666666666666666
Episode: 7	Step: 2100	Score: 20.0	Avg Score: 16.285714285714285
Episode: 8	Step: 2400	Score: 16.0	Avg Score: 16.25
Episode: 9	Step: 2700	Score: 16.0	Avg Score: 16.22222222222222
Episode: 10	Step: 3000	Score: 21.0	Avg Score: 16.7
Episode: 11	Step: 3300	Score: 18.0	Avg Score: 16.818181818181817
Episode: 12	Step: 3600	Score: 21.0	Avg Score: 17.166666666666668
Episode: 13	Step: 3900	Score: 20.0	Avg Score: 17.384615384615383
Episode: 14	Step: 4200	Score: 21.0	Avg Score: 17.642857142857142
Episode: 15	Step: 4500	Score: 19.0	Avg Score: 17.733333333333334
Episode: 16	Step: 4800	Score: 16.0	Avg Score: 17.625
Episode: 17	Step: 5100	Score: 14.0	Avg Score: 17.41176470588235
Episode: 18	Step: 5400	Score: 11.0	Avg Score: 17.055555555555557
Episode: 19	Step: 5700	Score: 12.0	Avg Score: 16.789473684210527
Episode: 20	Step: 6000	Score: 19.0	Avg Score: 16.9
Episode: 21	Step: 6300	Score: 19.0	Avg Score: 17.0
Episode: 22	Step: 6600	Score: 20.0	Avg Score: 17.136363636363637
Episode: 23	Step: 6900	Score: 17.0	Avg Score: 17.130434782608695
Episode: 24	Step: 7200	Score: 22.0	Avg Score: 17.333333333333332
Episode: 25	Step: 7500	Score: 12.0	Avg Score: 17.12
Episode: 26	Step: 7800	Score: 18.0	Avg Score: 17.153846153846153
Episode: 27	Step: 8100	Score: 21.0	Avg Score: 17.296296296296298
Episode: 28	Step: 8400	Score: 17.0	Avg Score: 17.285714285714285
Episode: 29	Step: 8700	Score: 15.0	Avg Score: 17.20689655172414
Episode: 30	Step: 9000	Score: 15.0	Avg Score: 17.133333333333333
Episode: 31	Step: 9300	Score: 25.0	Avg Score: 17.387096774193548
Episode: 32	Step: 9600	Score: 16.0	Avg Score: 17.34375
Episode: 33	Step: 9900	Score: 12.0	Avg Score: 17.181818181818183
Episode: 34	Step: 10200	Score: 21.0	Avg Score: 17.294117647058822
Episode: 35	Step: 10500	Score: 16.0	Avg Score: 17.257142857142856
Episode: 36	Step: 10800	Score: 21.0	Avg Score: 17.36111111111111
Episode: 37	Step: 11100	Score: 23.0	Avg Score: 17.513513513513512
Episode: 38	Step: 11400	Score: 20.0	Avg Score: 17.57894736842105
Episode: 39	Step: 11700	Score: 23.0	Avg Score: 17.71794871794872
Episode: 40	Step: 12000	Score: 20.0	Avg Score: 17.775
Episode: 41	Step: 12300	Score: 13.0	Avg Score: 17.658536585365855
Episode: 42	Step: 12600	Score: 21.0	Avg Score: 17.738095238095237
Episode: 43	Step: 12900	Score: 18.0	Avg Score: 17.74418604651163
Episode: 44	Step: 13200	Score: 21.0	Avg Score: 17.818181818181817
Episode: 45	Step: 13500	Score: 11.0	Avg Score: 17.666666666666668
Episode: 46	Step: 13800	Score: 19.0	Avg Score: 17.695652173913043
Episode: 47	Step: 14100	Score: 16.0	Avg Score: 17.659574468085108
Episode: 48	Step: 14400	Score: 20.0	Avg Score: 17.708333333333332
Episode: 49	Step: 14700	Score: 22.0	Avg Score: 17.79591836734694
Episode: 50	Step: 15000	Score: 24.0	Avg Score: 17.92
Episode: 51	Step: 15300	Score: 23.0	Avg Score: 18.019607843137255
Episode: 52	Step: 15600	Score: 18.0	Avg Score: 18.01923076923077
Episode: 53	Step: 15900	Score: 8.0	Avg Score: 17.830188679245282
Episode: 54	Step: 16200	Score: 18.0	Avg Score: 17.833333333333332
Episode: 55	Step: 16500	Score: 22.0	Avg Score: 17.90909090909091
Episode: 56	Step: 16800	Score: 17.0	Avg Score: 17.892857142857142
Episode: 57	Step: 17100	Score: 21.0	Avg Score: 17.94736842105263
Episode: 58	Step: 17400	Score: 19.0	Avg Score: 17.96551724137931
Episode: 59	Step: 17700	Score: 19.0	Avg Score: 17.983050847457626
Episode: 60	Step: 18000	Score: 20.0	Avg Score: 18.016666666666666
Episode: 61	Step: 18300	Score: 21.0	Avg Score: 18.065573770491802
Episode: 62	Step: 18600	Score: 18.0	Avg Score: 18.06451612903226
Episode: 63	Step: 18900	Score: 17.0	Avg Score: 18.047619047619047
Episode: 64	Step: 19200	Score: 15.0	Avg Score: 18.0
Episode: 65	Step: 19500	Score: 11.0	Avg Score: 17.892307692307693
Episode: 66	Step: 19800	Score: 21.0	Avg Score: 17.939393939393938
Episode: 67	Step: 20100	Score: 21.0	Avg Score: 17.98507462686567
Episode: 68	Step: 20400	Score: 13.0	Avg Score: 17.91176470588235
Episode: 69	Step: 20700	Score: 18.0	Avg Score: 17.91304347826087
Episode: 70	Step: 21000	Score: 26.0	Avg Score: 18.02857142857143
Episode: 71	Step: 21300	Score: 18.0	Avg Score: 18.028169014084508
Episode: 72	Step: 21600	Score: 17.0	Avg Score: 18.01388888888889
Episode: 73	Step: 21900	Score: 23.0	Avg Score: 18.08219178082192
Episode: 74	Step: 22200	Score: 15.0	Avg Score: 18.04054054054054
Episode: 75	Step: 22500	Score: 20.0	Avg Score: 18.066666666666666
Episode: 76	Step: 22800	Score: 20.0	Avg Score: 18.092105263157894
Episode: 77	Step: 23100	Score: 16.0	Avg Score: 18.064935064935064
Episode: 78	Step: 23400	Score: 22.0	Avg Score: 18.115384615384617
Episode: 79	Step: 23700	Score: 16.0	Avg Score: 18.088607594936708
Episode: 80	Step: 24000	Score: 28.0	Avg Score: 18.2125
Episode: 81	Step: 24300	Score: 16.0	Avg Score: 18.185185185185187
Episode: 82	Step: 24600	Score: 21.0	Avg Score: 18.21951219512195
Episode: 83	Step: 24900	Score: 19.0	Avg Score: 18.228915662650603
Episode: 84	Step: 25200	Score: 19.0	Avg Score: 18.238095238095237
Episode: 85	Step: 25500	Score: 14.0	Avg Score: 18.188235294117646
Episode: 86	Step: 25800	Score: 15.0	Avg Score: 18.151162790697676
Episode: 87	Step: 26100	Score: 20.0	Avg Score: 18.17241379310345
Episode: 88	Step: 26400	Score: 20.0	Avg Score: 18.193181818181817
Episode: 89	Step: 26700	Score: 13.0	Avg Score: 18.134831460674157
Episode: 90	Step: 27000	Score: 18.0	Avg Score: 18.133333333333333
Episode: 91	Step: 27300	Score: 18.0	Avg Score: 18.13186813186813
Episode: 92	Step: 27600	Score: 21.0	Avg Score: 18.16304347826087
Episode: 93	Step: 27900	Score: 20.0	Avg Score: 18.182795698924732
Episode: 94	Step: 28200	Score: 19.0	Avg Score: 18.19148936170213
Episode: 95	Step: 28500	Score: 19.0	Avg Score: 18.2
Episode: 96	Step: 28800	Score: 24.0	Avg Score: 18.260416666666668
Episode: 97	Step: 29100	Score: 1.0	Avg Score: 18.082474226804123
Episode: 98	Step: 29400	Score: 18.0	Avg Score: 18.081632653061224
Episode: 99	Step: 29700	Score: 19.0	Avg Score: 18.09090909090909
Episode: 100	Step: 30000	Score: 18.0	Avg Score: 18.09
```

Navigation RL nanodegree project using rainbow DQN
