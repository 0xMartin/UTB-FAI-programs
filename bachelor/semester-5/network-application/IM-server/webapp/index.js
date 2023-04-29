const webSocket = new WebSocket("ws://" + window.location.host + "/IMServer/chat");

// message
const message = document.querySelector("#message");

// user name
const name = document.querySelector("#name");

// chat tabs
const tab_chat = document.querySelector("#tab_chat");
const tab_chat_content = document.querySelector("#tab_chat_content");

// modal
const modal = document.querySelector("#myModal");
const modal_title = document.querySelector("#modal_title");
const modal_input = document.querySelector("#modal_input");
const modal_ok = document.querySelector("#modal_ok");
const modal_close = document.querySelector("#modal_close");


//INIT
//###################################################################################

webSocket.onopen = function(message) {
	wsOpen(message);
};
webSocket.onmessage = function(message) {
	wsGetMessage(message);
};
webSocket.onclose = function(message) {
	wsClose(message);
};
webSocket.onerror = function(message) {
	wsError(message);
};

//SERVER COMUNICATION
//###################################################################################
function wsOpen(message) {
	writeMsgToChat("", "", "Connected ...");
}

function wsSendMessage() {
	// odesle zpravu na server
	if (message.value.trim()[0] == '/') {
		// zprava je prikaz -> zacina /
		webSocket.send(message.value);

		//send user info request
		webSocket.send("/get_name");
		webSocket.send("/groups");
	} else {
		//normalni zprava -> automaticky posle prikaz pro odeslani zpravu do konkretni skupiny "/private_gr"
		var gr = getCurrentChatGroup();
		if (gr.length == 0) return;
		webSocket.send("/private_gr " + gr + " " + message.value);
	}
	message.value = "";
}

function wsCloseConnection() {
	webSocket.close();
}

function wsGetMessage(message) {
	console.log(message.data);
	const json_data = JSON.parse(message.data);
	if (json_data.type === "hidden") {
		// skryta zprava (ode serveru dostaneme info ve ktere skupine se uzivatel nachazi a jake ma jmeno)
		switch (json_data.data.name) {
			case "groups":
				generateChatTabs(json_data.data.value);
				print_groups(json_data.data.value);
				break;
			case "get_name":
				print_name(json_data.data.value);
				break;
		}
	} else {
		// viditelna zprava (normalne se vypise to textoveho pole)
		writeMsgToChat(json_data.group, json_data.user, json_data.data);
	}
}

function wsClose(message) {
	writeMsgToChat("", "", "Disconnect ...");
}

function wsError(message) {
	writeMsgToChat("", "", "Error ...");
}

function writeMsgToChat(group_name, user_name, message) {
	// pokud jmeno skupiny neni definovano pouzije "defaulni chat" = Personal
	group_name = group_name.length == 0 ? "Personal" : group_name;

	// vygeneruje html kod pro novou zpravu a prida jej do prislusneho chatu
	const chat = document.querySelector("#chat_" + group_name);
	if (chat == null) return;

	var str = '<li class="clearfix">';
	if (name.innerText == user_name) {
		// vlastni zprava
		str += '<div class="message-data text-end"><span class="message-data-time">' + user_name + '</span></div>';
		str += '<div class="message my-message float-right">' + message + '</div>';
	} else {
		// zprava od jineho uzivatele
		str += '<div class="message-data"><span class="message-data-time">' + user_name + '</span></div>';
		str += '<div class="message other-message">' + message + '</div>';
	}
	str += '</li>';

	chat.innerHTML += str;

	// otevre tab s chatem do ktereho dosla zprava
	document.querySelector("#btn_" + group_name).click();
	
	// scroll down
	var element = chat.parentElement.parentElement;
   	element.scrollTop = element.scrollHeight - element.clientHeight;
}

//INFO
//###################################################################################

function print_name(name_str) {
	name.innerHTML = name_str;
}

function print_groups(groups_str) {
	var str_out = "";
	groups_str.split(';').forEach(g => {
		str_out += "<div class='px-1'><span class='badge bg-secondary'>" + g + "</span></div>";
	});
	group_list.innerHTML = str_out;
}

//CHAT TABS
//###################################################################################

var current_groups = "";

function generateChatTabs(groups_str) {
	groups_str = "Personal;" + groups_str;
	
	// overi ktere skupiny byly naposledy sestaveny, pokude nedoslo k zadne zmene pak se generovani neprovede
	if(groups_str == current_groups) {
		return;	
	}
	current_groups = groups_str;
	
	
	var groups = groups_str.split(';');

	// zalohuje si predchozi text v chatech
	var data_map = new Map();
	groups.forEach(g => {
		var chat = document.querySelector("#chat_" + g);
		if (chat != null) {
			data_map.set(g, chat.innerHTML);
		}
	});

	// aktualne oznaceny tab
	var active_tab = groups[0];
	var c = getCurrentChatGroup();
	if (c.length != 0) {
		active_tab = c;
	}

	//vygenerovani headeru pro taby
	var str_buffer = "";

	groups.forEach(g => {
		if (g.length == 0) return;
		str_buffer += '<li class="nav-item" role="presentation">';
		if (g === active_tab) {
			str_buffer += '<button class="nav-link active" id="btn_' + g + '" data-bs-toggle="tab" data-bs-target="#' + g + '" type="button" role="tab">' + g + '</button>';
		} else {
			str_buffer += '<button class="nav-link" id="btn_' + g + '" data-bs-toggle="tab" data-bs-target="#' + g + '" type="button" role="tab">' + g + '</button>';
		}
		str_buffer += '</li>';
	});
	tab_chat.innerHTML = str_buffer;

	//vygeneruje content panel pro jednotlive taby
	str_buffer = "";
	groups.forEach(g => {
		if (g.length == 0) return;
		if (g === active_tab) {
			str_buffer += '<div class="tab-pane fade show active" id="' + g + '" role="tabpanel" >';
		} else {
			str_buffer += '<div class="tab-pane fade show" id="' + g + '" role="tabpanel" >';
		}

		str_buffer += '<div class="chat">';
		str_buffer += '<div class="chat-history">';
		str_buffer += '<ul class="m-b-0" id="chat_' + g + '">' + (data_map.get(g) ?? "") + '</ul>';
		str_buffer += '</div>';
		str_buffer += '</div>';

		str_buffer += '</div>';
	});
	tab_chat_content.innerHTML = str_buffer;
}

function getCurrentChatGroup() {
	var chat_ul = document.querySelector("div.active div.chat div.chat-history ul");
	if (chat_ul == null) return "";
	var group_name = chat_ul.id.substring(5);
	if (group_name == "Personal") return "";
	return group_name;
}

//MODAL
//###################################################################################

function modal_setName() {
	modal_title.innerHTML = "Type your new name";
	modal_input.value = "";
	modal_ok.onclick = function() {
		webSocket.send("/set_name " + modal_input.value);
		webSocket.send("/get_name");
		webSocket.send("/groups");
	};
}

function modal_joinGroup() {
	modal_title.innerHTML = "Type the name of the group you want to join";
	modal_input.value = "";
	modal_ok.onclick = function() {
		webSocket.send("/group_add " + modal_input.value);
		webSocket.send("/groups");
	};
}

function modal_leaveGroup() {
	modal_title.innerHTML = "Type the name of the group you want to leave";
	modal_input.value = "";
	modal_ok.onclick = function() {
		webSocket.send("/group_rmv " + modal_input.value);
		webSocket.send("/groups");
	};
}

function modal_sendPrivate() {
	modal_title.innerHTML = "Type the name of the receiver";
	modal_input.value = "";
	modal_ok.onclick = function() {
		webSocket.send("/private " + modal_input.value + " " + message.value);
		message.value = "";
	};
}

