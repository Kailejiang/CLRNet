import os

import ase
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations


import schnetpack.interfaces.ase_interface.AtomsConverter as AtomsConverter
from schnetpack.dataset import MD17

ethanol_ase = AseInterface(
    molecule_path,
    ase_dir,
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key=MD17.energy,
    force_key=MD17.forces,
    energy_unit="kcal/mol",
    position_unit="Ang",
    device="cpu",
    dtype=torch.float64,
)

ethanol_ase.init_md(
    'simulation'
)
ethanol_ase.run_md(1000)

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



class SpkCalculator(Calculator):
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(
        self,
        model_file: str,
        neighbor_list: schnetpack.transform.Transform,
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: callable = AtomsConverter,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        self.converter = converter(
            neighbor_list=neighbor_list,
            device=device,
            dtype=dtype,
            transforms=transforms,
            additional_inputs=additional_inputs,
        )

        self.energy_key = energy_key
        self.force_key = force_key

        # Mapping between ASE names and model outputs
        self.property_map = {
            self.energy: energy_key,
            self.forces: force_key,
        }

        self.model = self._load_model(model_file)
        self.model.to(device=device, dtype=dtype)

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy: self.energy_conversion,
            self.forces: self.energy_conversion / self.position_conversion,
        }

        # Container for basic ml model ouputs
        self.model_results = None

    def _load_model(self, model_file: str) -> schnetpack.model.AtomisticModel:

        log.info("Loading model from {:s}".format(model_file))
        # load model and keep it on CPU, device can be changed afterwards
        model = load_model(model_file, device=torch.device("cpu")).to(torch.float64)
        model = model.eval()

        return model

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)

            # Convert to schnetpack input format
            model_inputs = self.converter(atoms)
            model_results = self.model(model_inputs)

            results = {}
            for prop in properties:
                model_prop = self.property_map[prop]

                if model_prop in model_results:
                    if prop == self.energy:
                        # ase calculator should return scalar energy
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy().item()
                            * self.property_units[prop]
                        )
                    else:
                        results[prop] = (
                            model_results[model_prop].cpu().data.numpy()
                            * self.property_units[prop]
                        )
                else:
                    raise AtomsConverterError(
                        "'{:s}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(prop)
                    )

            self.results = results
            self.model_results = model_results


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
        energy_unit: Union[str, float] = "kcal/mol",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        converter: AtomsConverter = AtomsConverter,
        optimizer_class: type = QuasiNewton,
    ):
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = read(molecule_path)、

        # Set up optimizer
        self.optimizer_class = optimizer_class

        # Set up calculator
        calculator = SpkCalculator(
            model_file=model_file,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            device=device,
            dtype=dtype,
            converter=converter,
        )

        self.molecule.set_calculator(calculator)

        self.dynamics = None
        

    def init_md(
        self,
        name: str,
        time_step: float = 0.5,
        temp_init: float = 300,
        temp_bath: Optional[float] = None,
        reset: bool = False,
        interval: int = 1,
    ):

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
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)
