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

class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    """

    def __init__(
        self,
        molecule_path: str,
        working_dir: str,
        model_file: str,
        neighbor_list: schnetpack.transform.Transform,
        energy_key: str = "energy",
        force_key: str = "forces",
        stress_key: Optional[str] = None,
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: AtomsConverter = AtomsConverter,
        optimizer_class: type = QuasiNewton,
        fixed_atoms: Optional[List[int]] = None,
        transforms: Union[
            schnetpack.transform.Transform, List[schnetpack.transform.Transform]
        ] = None,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            molecule_path: Path to initial geometry
            working_dir: Path to directory where files should be stored
            model_file (str): path to trained model
            neighbor_list (schnetpack.transform.Transform): SchNetPack neighbor list
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            stress_key (str): name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit (str, float): energy units used by model (default="kcal/mol")
            position_unit (str, float): position units used by model (default="Angstrom")
            device (torch.device): device used for calculations (default="cpu")
            dtype (torch.dtype): select model precision (default=float32)
            converter (schnetpack.interfaces.AtomsConverter): converter used to set up input batches
            optimizer_class (ase.optimize.optimizer): ASE optimizer used for structure relaxation.
            fixed_atoms (list(int)): list of indices corresponding to atoms with positions fixed in space.
            transforms (schnetpack.transform.Transform, list): transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            additional_inputs (dict): additional inputs required for some transforms in the converter.
        """
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = read(molecule_path)

        # Apply position constraints
        if fixed_atoms:
            c = FixAtoms(fixed_atoms)
            self.molecule.set_constraint(constraint=c)

        # Set up optimizer
        self.optimizer_class = optimizer_class

        # Set up calculator
        calculator = SpkCalculator(
            model_file=model_file,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            device=device,
            dtype=dtype,
            converter=converter,
            transforms=transforms,
            additional_inputs=additional_inputs,
        )

        self.molecule.set_calculator(calculator)

        self.dynamics = None

    def save_molecule(self, name: str, file_format: str = "xyz", append: bool = False):
        """
        Save the current molecular geometry.

        Args:
            name: Name of save-file.
            file_format: Format to store geometry (default xyz).
            append: If set to true, geometry is added to end of file (default False).
        """
        molecule_path = os.path.join(
            self.working_dir, "{:s}.{:s}".format(name, file_format)
        )
        write(molecule_path, self.molecule, format=file_format, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule("single_point", file_format="xyz")

    def init_md(
        self,
        name: str,
        time_step: float = 0.5,
        temp_init: float = 300,
        temp_bath: Optional[float] = None,
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
                instead (default=None)
            reset: Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval: Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if self.dynamics is None or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
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

    def _init_velocities(
        self,
        temp_init: float = 300,
        remove_translation: bool = True,
        remove_rotation: bool = True,
    ):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init: Initial temperature in Kelvin (default 300)
            remove_translation: Remove translation components of velocity (default True)
            remove_rotation: Remove rotation components of velocity (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)
