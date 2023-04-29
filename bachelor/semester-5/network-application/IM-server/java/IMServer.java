/**
 * INFO:
 * 	Prikazy:
 * 		/set_name {jmeno} -> nastavy nove jmeno pro uzivatele
 * 		/get_name -> pro danou session zjisti uzivatelske jmeno
 * 		/group_add {jmeno_skupiny} -> prida uzivatele do skupiny, pokud skupina jeste neexistuje tak ji automaticky vytvori
 * 		/group_rmv {jmeno_skupiny} -> odebere uzivatele ze skupiny
 * 		/private {jmenu_prijemce} {zprava ...} -> odesle soukromou zpravu
 * 		/private_gr {jmenu_skupiny} {zprava ...} -> odesle soukromou zpravu (skupine)
 * 		/groups -> vypise seznam skupin, ve kterych je uzivatel
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import javax.websocket.*;
import javax.websocket.server.ServerEndpoint;

@ServerEndpoint("/chat")
public class IMServer {

	public static final String CMD_SET_NAME = "set_name";
	public static final String CMD_GET_NAME = "get_name";
	public static final String CMD_GROUP_ADD = "group_add";
	public static final String CMD_GROUP_RMV = "group_rmv";
	public static final String CMD_PRIVATE_MSG = "private";
	public static final String CMD_PRIVATE_GROUP_MSG = "private_gr";
	public static final String CMD_SHOW_GROUPS = "groups";

	public static final String DEFAULT_GROUP = "Default";

	public static final int GROUPS_LIMIT = 2;

	private static final ConcurrentHashMap<Session, User> USERS = new ConcurrentHashMap<Session, User>();
	private static final ConcurrentHashMap<String, Group> GROUPS = new ConcurrentHashMap<String, Group>();

	private static final IMServer.ChatObject SERVER_USER = new IMServer.User(null, "SERVER");

	@OnOpen
	public void onOpen(Session session) {
		System.out.println("Open Connection ... " + session.getId());
	}

	@OnClose
	public void onClose(Session session) {
		System.out.println("Close Connection ... " + session.getId());
		IMServer.User u = IMServer.USERS.get(session);
		// odebere uzivatele ze skupin ve kterych se nachazi
		if (u != null) {
			u.getGroups(IMServer.GROUPS.values()).stream().forEach(g -> {
				((Group) g).removeUser(u);
			});
		}
		// odebere uzivatele ze seznamu uzivatelu
		IMServer.USERS.remove(session);
	}

	@OnMessage
	public void onMessage(String message, Session session) {
		User from = IMServer.User.getUser(session);

		IMServer.Message mgs = parseCommand(from, message);
		if (mgs == null)
			return;
		if(mgs.message.length() == 0) 
			return;

		Iterator<IMServer.ChatObject> it = mgs.receivers.iterator();
		while (it.hasNext()) {
			try {
				it.next().sendMessage(mgs);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	@OnError
	public void onError(Throwable e) {
		e.printStackTrace();
	}

	public final IMServer.Message parseCommand(User user, String data) {
		data = data.trim();
		if (data.length() == 0)
			return null;

		if (data.charAt(0) == '/') {

			String command = "";
			for (int i = 1; i < data.length(); ++i) {
				if (data.charAt(i) == ' ') {
					data = data.substring(i);
					break;
				}
				command += data.charAt(i);
			}

			// prikazy
			String[] args = data.trim().split("\\s+");
			if (command.endsWith(CMD_SET_NAME)) {
				// nastaveni uzivatelskeho jmena
				if (user.setName(args[0])) {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("Name changed to [%s]", user.getName()));
				} else {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("Someone already has name [%s]", user.getName()));
				}

			} else if (command.endsWith(CMD_GET_NAME)) {
				// navraceni jmena uzivatele
				return new Message(MessegeType.HIDDEN_MSG, IMServer.SERVER_USER, user,
						IMServer.Message.dataWithName(CMD_GET_NAME, user.getName()));

			} else if (command.endsWith(CMD_GROUP_ADD)) {
				if (user.getGroups(IMServer.GROUPS.values()).size() >= IMServer.GROUPS_LIMIT) {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user, "Number of groups exceeded");
				}
				// pripojit se ke skupine
				Group g = IMServer.Group.getGroup(args[0]);
				if (g.addUser(user)) {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("You have joined to the group [%s]", args[0]));
				} else {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("you are already in this group [%s]", args[0]));
				}

			} else if (command.endsWith(CMD_GROUP_RMV)) {
				// odebere uzivatele ze skupiny
				Group g = IMServer.Group.getGroup(args[0]);
				if (g.removeUser(user)) {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("You have leaved the group [%s]", args[0]));
				} else {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("You are not in this group [%s]", args[0]));
				}

			} else if (command.endsWith(CMD_PRIVATE_MSG) && !user.getName().isEmpty()) {
				// odesle soukromou zpravu 
				
				List<IMServer.ChatObject> objects = new ArrayList<>(IMServer.USERS.values());
				Optional<IMServer.ChatObject> to = objects.stream().filter(u -> u.getName().equals(args[0]))
						.findFirst();
				if (!to.isEmpty()) {
					return new Message(MessegeType.TEXT_MSG, user, to.get(),
							String.join(" ", Arrays.copyOfRange(args, 1, args.length)));
				} else {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("User with name [%s] not found", args[0]));
				}

			} else if (command.endsWith(CMD_PRIVATE_GROUP_MSG) && !user.getName().isEmpty()) {
				// odesle soukromou zpravu (cele skupine)
				
				List<IMServer.ChatObject> objects = new ArrayList<>(IMServer.GROUPS.values());
				Optional<IMServer.ChatObject> to = objects.stream().filter(u -> u.getName().equals(args[0]))
						.findFirst();
				if (!to.isEmpty()) {
					return new Message(MessegeType.TEXT_MSG, user, to.get(),
							String.join(" ", Arrays.copyOfRange(args, 1, args.length)));
				} else {
					return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user,
							String.format("Group with name [%s] not found", args[0]));
				}

			} else if (command.endsWith(CMD_SHOW_GROUPS)) {
				// vypise skupine ve kterych se uzivatel nachazi
				String groups = "";
				for (IMServer.ChatObject co : user.getGroups(IMServer.GROUPS.values())) {
					groups += co.getName() + ";";
				}
				return new Message(MessegeType.HIDDEN_MSG, IMServer.SERVER_USER, user,
						IMServer.Message.dataWithName(CMD_SHOW_GROUPS, groups));

			} else {
				// neznamy prikaz
				return new Message(MessegeType.TEXT_MSG, IMServer.SERVER_USER, user, "Unknown command");
			}

		} else {
			// odesle zprava do vsech skupin ve kterych se uzivatel nachazi
			if (!user.getName().isEmpty()) {
				return new Message(MessegeType.TEXT_MSG, user, user.getGroups(IMServer.GROUPS.values()), data);
			} else {
				return null;
			}
		}
	}

	/****************************************************************************************************
	 * LOCAL CLASSES
	 ****************************************************************************************************/

	/**
	 * Trida uzivatele
	 */
	static class User extends IMServer.ChatObject {
		final Session session;
		String name;

		public User(Session session, String name) {
			this.session = session;
			this.name = name;
		}

		@Override
		public String getName() {
			return this.name;
		}

		@Override
		public Session getSession() {
			return this.session;
		}

		@Override
		public void sendMessage(Message msg) throws IOException {
			// if (msg.sender.getSession() == this.session) return;
			if (this.name.isEmpty())
				return;
			if (this.session.isOpen())

				this.session.getBasicRemote().sendText(msg.toString());
		}

		public boolean setName(String name) {
			boolean status = IMServer.USERS.values().stream().allMatch(u -> !u.name.equals(name));
			if (status) {
				this.name = name;
			}
			return status;
		}

		public List<ChatObject> getGroups(Collection<Group> groups) {
			return groups.stream().filter(g -> g.users.contains(this)).collect(Collectors.toList());
		}

		public final static User getUser(Session s) {
			User u = IMServer.USERS.get(s);
			if (u == null) {
				// vytvori noveho uzivatle
				u = new User(s, "");
				IMServer.Group.getGroup(IMServer.DEFAULT_GROUP).addUser(u);
				IMServer.USERS.put(s, u);
			}
			return u;
		}

	}

	/**
	 * Trida skupiny
	 */
	static class Group extends IMServer.ChatObject {
		public List<User> users;
		public final String name;

		public Group(String name) {
			this.name = name;
			this.users = Collections.synchronizedList(new ArrayList<User>());
		}

		public boolean addUser(User u) {
			if (!users.contains(u)) {
				users.add(u);
				return true;
			}
			return false;
		}

		public boolean removeUser(User u) {
			return this.users.remove(u);
		}

		@Override
		public String getName() {
			return this.name;
		}

		@Override
		public void sendMessage(Message msg) throws IOException {
			msg.group = this.name;
			this.users.stream().forEach(u -> {
				try {
					u.sendMessage(msg);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}

		public final static Group getGroup(String name) {
			Group g = IMServer.GROUPS.get(name);
			if (g == null) {
				g = new Group(name);
				IMServer.GROUPS.put(name, g);
			}
			return g;
		}

	}

	/**
	 * Interface ChatObject
	 */
	static abstract class ChatObject {

		public String getName() {
			return "";
		}

		public Session getSession() {
			return null;
		}

		public abstract void sendMessage(Message msg) throws IOException;
	}

	/**
	 * Type zpravy
	 */
	enum MessegeType {
		TEXT_MSG("text"), HIDDEN_MSG("hidden");

		final String txt;

		MessegeType(String txt) {
			this.txt = txt;
		}

		@Override
		public String toString() {
			return this.txt;
		}
	}

	/**
	 * Zprava
	 */
	static class Message {

		final MessegeType type;
		final ChatObject sender;
		final List<ChatObject> receivers;
		String message;
		String group;

		public Message(MessegeType type, ChatObject sender, List<ChatObject> receivers, String message) {
			this.type = type;
			this.sender = sender;
			this.receivers = receivers;
			this.message = message;
			this.group = "";
		}

		public Message(MessegeType type, ChatObject sender, ChatObject receiver, String message) {
			this.type = type;
			this.sender = sender;
			this.receivers = new ArrayList<ChatObject>();
			this.receivers.add(receiver);
			this.message = message;
			this.group = "";
		}

		@Override
		public String toString() {
			String out = this.message;
			if (this.type == MessegeType.TEXT_MSG) {
				out = "\"" + out + "\"";
			}
			return String.format("{\"type\":\"%s\", \"user\":\"%s\", \"group\":\"%s\", \"data\":%s}", this.type.toString(), sender.getName(), group, out);
		}

		public static final String dataWithName(String name, String data) {
			return String.format("{\"name\":\"%s\", \"value\":\"%s\"}", name, data);
		}

	}

}
