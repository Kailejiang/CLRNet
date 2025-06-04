from ase.md import Langevin, MDLogger
from ase.io.trajectory import Trajectory

# Load logged results
results = np.loadtxt(os.path.join(ase_dir, 'simulation.log'), skiprows=1)

# Determine time axis
time = results[:,0]

# Load energies
energy_tot = results[:,1]
energy_pot = results[:,2]
energy_kin = results[:,3]

# Construct figure
plt.figure(figsize=(14,6))

# Plot energies
plt.subplot(2,1,1)
plt.plot(time, energy_tot, label='Total energy')
plt.plot(time, energy_pot, label='Potential energy')
plt.ylabel('E [eV]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time, energy_kin, label='Kinetic energy')
plt.ylabel('E [eV]')
plt.xlabel('Time [ps]')
plt.legend()

temperature = results[:,4]
print('Average temperature: {:10.2f} K'.format(np.mean(temperature)))

plt.show()

    def init_md(
        self,
        name: str,
        time_step: float = 0.5,
        temp_init: float = 300,
        temp_bath: Optional[float] = 300,
        reset: bool = False,
        interval: int = 1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.

        Args:
            name: Basic name of logfile and trajectory
            time_step: Time step in fs (default=0.5)
            temp_init: Initial temperature of the system in K (default is 300)
            temp_bath: Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)300
            reset: Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval: Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if self.dynamics is None or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        self.dynamics = Langevin(
            self.molecule,
            time_step * units.fs,
            temp_bath * units.kB,
            1.0 / (100.0 * units.fs),
        )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "{:s}.log".format(name))
        trajfile = os.path.join(self.working_dir, "{:s}.traj".format(name))
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)
