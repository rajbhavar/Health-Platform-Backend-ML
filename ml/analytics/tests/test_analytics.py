import csv
import tempfile
import unittest

from pathlib import Path

from analytics import calculate


class TestAnalytics(unittest.TestCase):

    def write_csv(self, folder, filename, headers, rows):
        path = Path(folder) / filename

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

    def create_dataset(
        self,
        folder,
        emergencies=None,
        ambulances=None,
        hospitals=None,
    ):
        self.write_csv(
            folder,
            "emergencies.csv",
            [
                "emergency_id",
                "reported_at",
                "emergency_type",
                "response_minutes",
            ],
            emergencies or [],
        )

        self.write_csv(
            folder,
            "ambulances.csv",
            [
                "ambulance_id",
                "status",
            ],
            ambulances or [],
        )

        self.write_csv(
            folder,
            "hospitals.csv",
            [
                "hospital_id",
                "status",
                "icu_beds_available",
            ],
            hospitals or [],
        )

    def test_empty_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            self.create_dataset(temp)

            summary, details = calculate(
                temp,
                "2026-10-13",
            )

            self.assertEqual(
                summary["emergencyCountToday"],
                0,
            )

            self.assertEqual(
                summary["avgResponseMinutes"],
                0.0,
            )

            self.assertEqual(
                len(summary["emergenciesByHour"]),
                24,
            )

            self.assertEqual(
                details["weeklyEmergencyCount"],
                0,
            )

    def test_valid_dashboard_calculations(self):
        with tempfile.TemporaryDirectory() as temp:
            self.create_dataset(
                temp,
                emergencies=[
                    [
                        "E1",
                        "2026-10-13T09:00:00Z",
                        "Medical",
                        "10",
                    ],
                    [
                        "E2",
                        "2026-10-13T09:30:00Z",
                        "Accident",
                        "20",
                    ],
                    [
                        "E3",
                        "2026-10-12T10:00:00Z",
                        "Medical",
                        "30",
                    ],
                ],
                ambulances=[
                    ["A1", "ACTIVE"],
                    ["A2", " active "],
                    ["A3", "BUSY"],
                ],
                hospitals=[
                    ["H1", "OPEN", "5"],
                    ["H2", "OPEN", "1"],
                    ["H3", "CLOSED", "4"],
                ],
            )

            summary, details = calculate(
                temp,
                "2026-10-13",
            )

            self.assertEqual(
                summary["emergencyCountToday"],
                2,
            )

            self.assertEqual(
                summary["activeAmbulances"],
                2,
            )

            self.assertEqual(
                summary["hospitalsAvailable"],
                2,
            )

            self.assertEqual(
                summary["avgResponseMinutes"],
                15.0,
            )

            self.assertEqual(
                summary["emergenciesByHour"][9]["count"],
                2,
            )

            self.assertEqual(
                details["weeklyEmergencyCount"],
                3,
            )

            self.assertEqual(
                len(details["hourlyDemandTrendEstimate"]),
                24,
            )

            self.assertTrue(
                all(
                    item["isEstimate"]
                    for item in details[
                        "hourlyDemandTrendEstimate"
                    ]
                )
            )

    def test_invalid_values_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            self.create_dataset(
                temp,
                emergencies=[
                    [
                        "E1",
                        "2026-10-13T10:00:00Z",
                        "Medical",
                        "-5",
                    ],
                    [
                        "E2",
                        "2026-10-13T11:00:00Z",
                        "Medical",
                        "300",
                    ],
                    [
                        "E3",
                        "invalid-date",
                        "Medical",
                        "10",
                    ],
                ],
                ambulances=[
                    ["A1", "UNKNOWN"],
                ],
                hospitals=[
                    ["H1", "OPEN", ""],
                    ["H2", "OPEN", "-2"],
                    ["H3", "OPEN", "1.5"],
                ],
            )

            summary, details = calculate(
                temp,
                "2026-10-13",
            )

            self.assertEqual(
                summary["emergencyCountToday"],
                2,
            )

            self.assertEqual(
                summary["avgResponseMinutes"],
                0.0,
            )

            self.assertEqual(
                summary["hospitalsAvailable"],
                0,
            )

            self.assertEqual(
                details["weeklyEmergencyCount"],
                2,
            )

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                calculate(
                    temp,
                    "2026-10-13",
                )

    def test_invalid_date(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                calculate(
                    temp,
                    "not-a-date",
                )

    def test_missing_required_column(self):
        with tempfile.TemporaryDirectory() as temp:
            self.create_dataset(temp)

            self.write_csv(
                temp,
                "emergencies.csv",
                ["emergency_id"],
                [["E1"]],
            )

            with self.assertRaises(ValueError):
                calculate(
                    temp,
                    "2026-10-13",
                )


if __name__ == "__main__":
    unittest.main()
