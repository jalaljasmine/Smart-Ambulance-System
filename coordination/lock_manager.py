import os
import pandas as pd
from typing import Dict, Optional

class HospitalResourcePool:
    """
    Tracks the resource pools (specifically beds) for each hospital.
    """
    def __init__(self, hospital_id: str, total_beds: int):
        self.hospital_id: str = hospital_id
        self.total_beds: int = total_beds
        self.locked_beds: int = 0
        self.occupied_beds: int = 0

    @property
    def available_beds(self) -> int:
        """
        Derived property showing currently available beds (not locked or occupied).
        """
        return self.total_beds - self.locked_beds - self.occupied_beds

    def __repr__(self) -> str:
        return (f"HospitalResourcePool(hospital_id='{self.hospital_id}', "
                f"total={self.total_beds}, locked={self.locked_beds}, "
                f"occupied={self.occupied_beds}, available={self.available_beds})")


class ReservationLock:
    """
    Represents a bed reservation lock held by an ambulance.
    """
    def __init__(self, lock_id: str, hospital_id: str, ambulance_id: str, 
                 claimed_at: float, expected_arrival: float, expires_at: float, 
                 status: str = "HELD"):
        self.lock_id: str = lock_id
        self.hospital_id: str = hospital_id
        self.ambulance_id: str = ambulance_id
        self.claimed_at: float = claimed_at
        self.expected_arrival: float = expected_arrival
        self.expires_at: float = expires_at
        self.status: str = status  # HELD, RELEASED, CONVERTED, EXPIRED

    def __repr__(self) -> str:
        return (f"ReservationLock(lock_id='{self.lock_id}', hospital='{self.hospital_id}', "
                f"ambulance='{self.ambulance_id}', status='{self.status}', "
                f"expires_at={self.expires_at:.2f})")


class LockManager:
    """
    Coordinates reservation locks and resource pools in-memory.
    Can be swapped later for a SQLite or Redis implementation.
    """
    def __init__(self, csv_path: str):
        self.pools: Dict[str, HospitalResourcePool] = {}
        self.locks: Dict[str, ReservationLock] = {}
        self.lock_counter: int = 0
        self._load_pools(csv_path)

    def _load_pools(self, csv_path: str) -> None:
        """Initializes resource pools from datasets/hospitals.csv."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Hospitals file not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            hospital_name = str(row["Hospital"])
            beds = int(row["ICU_Beds"])
            self.pools[hospital_name] = HospitalResourcePool(hospital_name, beds)

    def claim_lock(self, hospital_id: str, ambulance_id: str, 
                   expected_arrival: float, current_time: float, 
                   buffer: float = 8.0) -> Optional[ReservationLock]:
        """
        Attempts to atomically reserve a bed at the specified hospital.
        Increments locked_beds only if available_beds > 0.
        """
        pool = self.pools.get(hospital_id)
        if not pool or pool.available_beds <= 0:
            return None

        self.lock_counter += 1
        lock_id = f"lock_{self.lock_counter}"
        expires_at = expected_arrival + buffer

        lock = ReservationLock(
            lock_id=lock_id,
            hospital_id=hospital_id,
            ambulance_id=ambulance_id,
            claimed_at=current_time,
            expected_arrival=expected_arrival,
            expires_at=expires_at,
            status="HELD"
        )
        self.locks[lock_id] = lock
        pool.locked_beds += 1
        return lock

    def release_lock(self, lock_id: str) -> bool:
        """
        Releases a reservation or occupation, reclaiming the bed.
        """
        lock = self.locks.get(lock_id)
        if not lock:
            return False

        pool = self.pools.get(lock.hospital_id)
        if not pool:
            return False

        if lock.status == "HELD":
            pool.locked_beds = max(0, pool.locked_beds - 1)
            lock.status = "RELEASED"
            return True
        elif lock.status == "CONVERTED":
            pool.occupied_beds = max(0, pool.occupied_beds - 1)
            lock.status = "RELEASED"
            return True
        elif lock.status == "EXPIRED":
            # If it already expired, locked_beds was already decremented during expiration.
            lock.status = "RELEASED"
            return True
        return False

    def convert_to_occupied(self, lock_id: str) -> bool:
        """
        Converts a HELD lock to CONVERTED (occupied) when the ambulance arrives.
        Decrements locked_beds and increments occupied_beds.
        """
        lock = self.locks.get(lock_id)
        if not lock or lock.status != "HELD":
            return False

        pool = self.pools.get(lock.hospital_id)
        if not pool:
            return False

        pool.locked_beds = max(0, pool.locked_beds - 1)
        pool.occupied_beds += 1
        lock.status = "CONVERTED"
        return True

    def expire_stale_locks(self, current_time: float) -> int:
        """
        Checks all HELD locks and marks them as EXPIRED if current_time > expires_at.
        Decrements locked_beds for any expired locks.
        Returns the number of expired locks.
        """
        expired_count = 0
        for lock in self.locks.values():
            if lock.status == "HELD" and current_time > lock.expires_at:
                lock.status = "EXPIRED"
                pool = self.pools.get(lock.hospital_id)
                if pool:
                    pool.locked_beds = max(0, pool.locked_beds - 1)
                expired_count += 1
        return expired_count
