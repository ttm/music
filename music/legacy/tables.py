"""Legacy lookup tables for common waveforms."""

from ..tables import PrimaryTables


class Basic(PrimaryTables):
    """Primary waveform tables, under the name the legacy synthesizers use.

    This was a third copy of the same four table definitions. It is now
    :class:`music.tables.PrimaryTables` under its historical name, so
    `CanonicalSynth` and anything else reaching for
    ``music.legacy.tables.Basic`` keeps working.

    Parameters
    ----------
    size : int, optional
        The number of samples for each waveform table, by default 2048.

    See Also
    --------
    music.tables.PrimaryTables : the class this is an alias for.
    music.utils.waveform_table : builds one table.
    """
