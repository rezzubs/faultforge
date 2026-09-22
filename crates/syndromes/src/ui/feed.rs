//! Where the snapshot on screen comes from.

use crate::ui::snapshot::Snapshot;

/// The source of the primary snapshot.
///
/// The app asks it for a newer snapshot once per frame and otherwise knows
/// nothing about where the data comes from.
#[derive(Debug)]
pub enum Feed {
    /// A saved file, which yielded its snapshot when it was loaded.
    File,
}

impl Feed {
    /// The next snapshot, if a newer one is ready.
    ///
    /// `None` means the one already on screen still stands.
    pub fn poll(&mut self) -> Option<Snapshot> {
        match self {
            Self::File => None,
        }
    }

    /// Whether more snapshots may still arrive.
    ///
    /// The app only asks for repaints while this holds.
    pub fn is_live(&self) -> bool {
        match self {
            Self::File => false,
        }
    }
}
