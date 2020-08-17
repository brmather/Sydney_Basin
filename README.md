# Sydney Basin model

The thermal and hydraulic regime is solved over a geological model of the Sydney Basin. The Sydney Basin model brings together thermal and hydraulic data to constrain numerical simulations.

#### Dependencies

- [`underworld2`](https://github.com/underworldcode/underworld2)
- [`stripy`](https://github.com/underworldcode/stripy)

## Geological model

The following geological surfaces are stored in the `Data` directory:

1. Elevation
2. Maules (top)
3. Maules (bottom)
4. Jurassic coal (top)
5. Jurassic coal (bottom)
6. Jurassic (bottom)
7. Greta CM (top)
8. Greta CM (bottom)
9. PCM (top)
10. Reid Dome Beds (top)
11. Reid Dome Beds (bottom)
12. PCM (bottom)
13. Denison Volcanics (top)
14. Denison Volcanics (bottom)
15. Complete volcanics onshore
16. Complete volcanics offshore
17. Complete basement


## Bayesian probability

Model averaging possible for the vector flows below if you wanted to? Would require:

- Saving vector flows for each posterior combination of parameters,
- Adding them together (could do cumulatively, multiplied my likelihood at each sample to save memory)
- Dividing by the right normalisation constant at the end.
- To save compute this could be done only every x = N_samples/n_eff samples, where n_eff is the effective sample size, i.e. the number of statistically independent samples.
- I would still then look at uncertainties the same way you would for the MAP estimate.

But maybe just keep MAP estimate for _Scientific Reports_!


## Acknowledgements

The Sydney Basin model was provided by Dr. Craig O'Neill of Macquarie University. The geological model was constructed by Dr. Cara Danis.

This work was made possible by the NSW Department of Industry from The Office of the Chief Scientist and Engineer and awarded to AuScope Simulations Analysis and Modelling.