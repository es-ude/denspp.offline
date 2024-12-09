"""Module providing a simple memristance curve for testing."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Memristance:
    """Representation of the Memristance Feature of a Memristor.

    Args:
        positive_read (npt.NDArray[np.float64]):
            array holding the data of the postiive read branch
            Column 1 = voltage, Column 2 = current
        negative_read (npt.NDArray[np.float64]):
            array holding the data of the negative read branch
            Column 1 = voltage, Column 2 = current
        positive_write (npt.NDArray[np.float64]):
            array holding the data of the postiive write branch
            Column 1 = voltage, Column 2 = current
        negative_write (npt.NDArray[np.float64]):
            array holding the data of the postiive write branch
            Column 1 = voltage, Column 2 = current
    """

    __positive_read: npt.NDArray[np.float64]
    __negative_read: npt.NDArray[np.float64]
    __positive_write: npt.NDArray[np.float64]
    __negative_write: npt.NDArray[np.float64]

    def __init__(
        self,
        positive_read: npt.NDArray[np.float64],
        negative_read: npt.NDArray[np.float64],
        positive_write: npt.NDArray[np.float64],
        negative_write: npt.NDArray[np.float64],
    ) -> None:
        """Create instance of Memristance data-class."""
        self.__positive_read = positive_read
        self.__negative_read = negative_read
        self.__positive_write = positive_write
        self.__negative_write = negative_write

    def get_current(self, voltage: float) -> float:
        """Get the current for a given voltage.

        Args:
            voltage (float): applied voltage over Memristor

        Returns:
            current (float): appropiate current of read branch
        """
        if voltage >= 0:
            val = np.interp(
                [voltage], self.__positive_read[:, 0], self.__positive_read[:, 1]
            )
            return val[0]
        else:
            val = np.interp(
                [voltage], self.__negative_read[:, 0], self.__negative_read[:, 1]
            )
            return val[0]

    def show_graph(self, blocking: bool = True, show_grid: bool = False) -> None:
        """Showing memristance.

        Args:
            blocking (bool): if this function should block when called
            show_grid (bool): show grid in graph (default=False)
        """
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current [Ω]")
        if show_grid:
            plt.grid(linestyle="dotted", color="black", linewidth=0.5)
        else:
            plt.axhline(linestyle="dotted", color="black", linewidth=1)
            plt.axvline(linestyle="dotted", color="black", linewidth=1)

        plt.plot(
            self.__positive_write[:, 0],
            self.__positive_write[:, 1],
            "-",
            label="WRITE (Branch 1)",
        )
        plt.plot(
            self.__positive_read[:, 0],
            self.__positive_read[:, 1],
            "-",
            label="READ (Branch 2)",
        )
        plt.plot(
            self.__negative_write[:, 0],
            self.__negative_write[:, 1],
            "-",
            label="WRITE (Branch 3)",
        )
        plt.plot(
            self.__negative_read[:, 0],
            self.__negative_read[:, 1],
            "-",
            label="READ (Branch 4)",
        )

        plt.legend(loc="best")
        plt.show(block=True)

    def get_branch(self, id: int) -> npt.NDArray[np.float64]:
        """Get Datapoints for selected Branch.

        Args:
            id (int): id of the branch
                      1 = Positive Write Branch
                      2 = Positive Read Branch
                      3 = Negative Write Branch
                      4 = Negative ReaNegative Read

        Returns:
            branch (npt.NDArray[np.float64):
                Array holding the data points for the requested branch
                Column 1 = voltage, Column 2 = current
        """
        if id == 1:
            return self.__positive_write
        elif id == 2:
            return self.__positive_read
        elif id == 3:
            return self.__negative_write
        elif id == 4:
            return self.__negative_read
        else:
            return None


def generate_sample_curve(max_voltage: float = 3, precision: int = 1000) -> Memristance:
    """Generate a simple curve for testing purpuses.

    Args:
        max_voltage (float): Lower and Upper bound for x-axis (default=3)
        precision (int): Number of values to calculate for each branch (default=1000)

    Returns:
        memristance (Memristance): New Instance of class Memristance with sample data
    """
    write_data = np.zeros([precision * 2, 2])
    read_data = np.zeros([precision * 2, 2])

    write_data[:, 0] = np.linspace(-max_voltage, max_voltage, precision * 2)
    read_data[:, 0] = np.linspace(-max_voltage, max_voltage, precision * 2)

    write_data[:, 1] = (max_voltage / (max_voltage**9)) * (write_data[:, 0] ** 9)
    read_data[precision:, 1] = (max_voltage / (max_voltage**2)) * (
        read_data[precision:, 0] ** 2
    )
    read_data[:precision, 1] = (
        (-1) * (max_voltage / (max_voltage**2)) * (read_data[:precision, 0] ** 2)
    )
    memristance = Memristance(
        read_data[precision:, :],
        read_data[:precision, :],
        write_data[precision:, :],
        write_data[:precision, :],
    )
    return memristance


if __name__ == "__main__":
    memristance = generate_sample_curve(max_voltage=12)
    memristance.show_graph(show_grid=True)
