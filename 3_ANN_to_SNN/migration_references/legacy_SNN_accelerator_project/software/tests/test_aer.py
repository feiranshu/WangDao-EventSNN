import unittest

from snn_accel.aer import PacketType, pack_packet, unpack_packet


class AerPacketTests(unittest.TestCase):
    def test_round_trip_maximum_event(self):
        packet = pack_packet(PacketType.EVENT, x=319, y=255, polarity=1, time_bin=15, slice_id=65535)
        decoded = unpack_packet(packet)
        self.assertEqual(decoded.packet_type, PacketType.EVENT)
        self.assertEqual(decoded.x, 319)
        self.assertEqual(decoded.y, 255)
        self.assertEqual(decoded.polarity, 1)
        self.assertEqual(decoded.time_bin, 15)
        self.assertEqual(decoded.slice_id, 65535)
        self.assertEqual(decoded.reserved, 0)

    def test_rejects_out_of_range_and_reserved_bits(self):
        with self.assertRaises(ValueError):
            pack_packet(PacketType.EVENT, x=320)
        with self.assertRaises(ValueError):
            unpack_packet(1 << 63)


if __name__ == "__main__":
    unittest.main()

