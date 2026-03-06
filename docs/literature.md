# Literature Review

This project sits at the intersection of Markov games, robust multi-agent learning, probabilistic forecast evaluation, and forecast aggregation. The list below is intentionally practical: each citation is included because it informs a concrete design choice in the repository.

## Design Links To Prior Work

1. **Markov games for adversarial forecasting**
   Littman formalized Markov games as the extension of MDPs to multi-agent settings. That framing matches the core game loop in this repository, where forecasters, adversaries, and defenders optimize interacting objectives rather than a single policy in isolation.

2. **Adaptive MARL learning rates**
   WoLF-PHC motivates the repository's equilibrium-seeking training mode. The key idea is to learn cautiously when winning and aggressively when losing, which is a natural fit for non-stationary adversarial forecasting environments.

3. **Robust adversarial RL**
   RARL motivates treating disturbances as an explicit opposing policy instead of as passive noise. In this codebase, disturbances and attack costs are first-class parts of the simulation rather than afterthought perturbations.

4. **Kelly-style allocation and forecast pooling**
   The Bayesian model averaging path uses bankroll-style updates that are philosophically close to Kelly allocation: capital is reallocated toward better calibrated experts over repeated rounds.

5. **Proper scoring rules**
   CRPS and related calibration metrics are grounded in proper scoring-rule literature. That matters because robust forecasting should optimize not only point accuracy but also distributional quality.

6. **Forecast combination**
   Bates and Granger remain the classic justification for combining multiple imperfect forecasters. The ensemble and BMA components are the practical implementation of that argument inside the game.

7. **Baseline forecasting**
   ARIMA-style baselines remain necessary in academic evaluation even for learning systems. This repo's benchmark harness is designed so classical baselines can be compared against the MARL system under the same rolling-window protocol.

8. **Scalable forecasting baselines**
   Prophet is not the main method here, but it is a standard industrial baseline and is referenced so benchmark protocols remain legible to readers outside reinforcement learning.

9. **Online learning and regret**
   The bandit backends are informed by online-learning theory. For stochastic-gap settings, UCB-style behavior motivates logarithmic regret discussions; for adversarial settings, Tsallis-INF provides a stronger theoretical reference point.

10. **Sequential prediction under uncertainty**
    Prediction-with-expert-advice literature is relevant because the forecaster set, ensemble updates, and adversarial perturbations can all be interpreted through sequential allocation under uncertain expert quality.

## Bibliography

```bibtex
@inproceedings{littman1994markov,
  title={Markov Games as a Framework for Multi-Agent Reinforcement Learning},
  author={Littman, Michael L.},
  booktitle={Proceedings of the 11th International Conference on Machine Learning},
  year={1994},
  url={https://www.cs.rutgers.edu/~mlittman/papers/ml94-final.pdf}
}

@article{bowling2002wolf,
  title={Multiagent Learning Using a Variable Learning Rate},
  author={Bowling, Michael and Veloso, Manuela},
  journal={Artificial Intelligence},
  volume={136},
  number={2},
  pages={215--250},
  year={2002},
  url={https://www.cs.cmu.edu/~mhb/papers/02aij.pdf}
}

@article{pinto2017rarl,
  title={Robust Adversarial Reinforcement Learning},
  author={Pinto, Lerrel and Davidson, James and Sukthankar, Rahul and Gupta, Abhinav},
  journal={arXiv preprint arXiv:1703.02702},
  year={2017},
  url={https://arxiv.org/abs/1703.02702}
}

@article{kelly1956information,
  title={A New Interpretation of Information Rate},
  author={Kelly, J. L.},
  journal={Bell System Technical Journal},
  volume={35},
  number={4},
  pages={917--926},
  year={1956},
  doi={10.1002/j.1538-7305.1956.tb03809.x}
}

@article{bates1969combination,
  title={The Combination of Forecasts},
  author={Bates, J. M. and Granger, Clive W. J.},
  journal={OR},
  volume={20},
  number={4},
  pages={451--468},
  year={1969},
  doi={10.1057/jors.1969.103}
}

@article{gneiting2007scoring,
  title={Strictly Proper Scoring Rules, Prediction, and Estimation},
  author={Gneiting, Tilmann and Raftery, Adrian E.},
  journal={Journal of the American Statistical Association},
  volume={102},
  number={477},
  pages={359--378},
  year={2007},
  doi={10.1198/016214506000001437}
}

@article{hyndman2008automatic,
  title={Automatic Time Series Forecasting: The forecast Package for R},
  author={Hyndman, Rob J. and Khandakar, Yeasmin},
  journal={Journal of Statistical Software},
  volume={27},
  number={3},
  pages={1--22},
  year={2008},
  url={https://www.jstatsoft.org/article/view/v027i03}
}

@article{taylor2018prophet,
  title={Forecasting at Scale},
  author={Taylor, Sean J. and Letham, Benjamin},
  journal={The American Statistician},
  volume={72},
  number={1},
  pages={37--45},
  year={2018},
  url={https://peerj.com/preprints/3190/}
}

@article{zimmert2019tsallis,
  title={Tsallis-INF: An Optimal Algorithm for Stochastic and Adversarial Bandits},
  author={Zimmert, Julian and Seldin, Yevgeny},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={28},
  pages={1--49},
  year={2021},
  url={https://jmlr.org/papers/v22/19-753.html}
}

@book{cesa2006prediction,
  title={Prediction, Learning, and Games},
  author={Cesa-Bianchi, Nicolo and Lugosi, Gabor},
  year={2006},
  publisher={Cambridge University Press},
  url={https://www.cambridge.org/core/books/prediction-learning-and-games/2465A6E76EA8028FA8C4E8F5B2A0E0F2}
}

@book{sutton2018rl,
  title={Reinforcement Learning: An Introduction},
  author={Sutton, Richard S. and Barto, Andrew G.},
  edition={2},
  year={2018},
  publisher={MIT Press},
  url={http://incompleteideas.net/book/the-book-2nd.html}
}
```

## Notes For Future Expansion

- Add explicit citations for any future deep-RL baseline such as PPO, DQN, or recurrent sequence models once those baselines become part of the default benchmark suite.
- Add a data ethics subsection tied to each real-world adapter when source-specific limitations are documented in greater depth.
