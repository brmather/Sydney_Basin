# Sydney Basin model

The thermal and hydraulic regime is solved over a geological model of the Sydney Basin. The Sydney Basin model brings together thermal and hydraulic data to constrain numerical simulations.

#### Dependencies

- [`stripy`](https://github.com/underworldcode/stripy)
- [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/install.html)
- [`h5py`](http://docs.h5py.org/en/stable/quick.html) (parallel-enabled)

## Geological model

The following geological surfaces are stored in the `Data` directory:

1. Elevation
2. Maules (top)
3. Maules (bottom)
4. Jurassic (bottom)
5. Jurassic coal (top)
6. Jurassic coal (bottom)
7. Greta CM (top)
8. Greta CM (bottom)
9. PCM (top)
10. PCM (bottom)
11. Reid Dome Beds (top)
12. Reid Dome Beds (bottom)
13. Denison Volcanics (top)
14. Denison Volcanics (bottom)
15. Complete volcanics onshore
16. Complete volcanics offshore
17. Complete basement

## Acknowledgements

The Sydney Basin model was provided by Dr. Craig O'Neill of Macquarie University. The geological model was constructed by Dr. Cara Danis.

This work was made possible by the NSW Department of Industry from The Office of the Chief Scientist and Engineer and awarded to AuScope Simulations Analysis and Modelling.