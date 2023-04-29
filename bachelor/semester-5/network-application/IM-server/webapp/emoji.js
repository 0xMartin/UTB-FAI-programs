
// https://github.com/Guad/simple-emoji-picker

function emojipicker(options) {
	function insertAtCaret(el, text) {
		text = text || "";
		if (document.selection) {
			// IE
			this.focus();
			var sel = document.selection.createRange();
			sel.text = text;
		} else if (el.selectionStart || el.selectionStart === 0) {
			// Others
			var startPos = el.selectionStart;
			var endPos = el.selectionEnd;
			el.value =
				el.value.substring(0, startPos) +
				text +
				el.value.substring(endPos, el.value.length);
			el.selectionStart = startPos + text.length;
			el.selectionEnd = startPos + text.length;
		} else {
			el.value += text;
		}
	}
	function hideOnClickOutside(element, onclick) {
		const outsideClickListener = event => {
			if (!element.contains(event.target) && isVisible(element)) {
				// or use: event.target.closest(selector) === null
				onclick();
				removeClickListener();
			}
		};

		const removeClickListener = () => {
			document.removeEventListener("click", outsideClickListener);
		};

		document.addEventListener("click", outsideClickListener);
	}

	const isVisible = elem =>
		!!elem &&
		!!(elem.offsetWidth || elem.offsetHeight || elem.getClientRects().length); // source (2018-03-11): https://github.com/jquery/jquery/blob/master/src/css/hiddenVisibleSelectors.js

	function emojiBtn(emoji, textarea) {
		var btn = document.createElement("span");
		btn.className = "emoji-btn";
		btn.innerText = emoji;
		btn.onclick = function() {
			insertAtCaret(textarea, emoji);
		};

		return btn;
	}

	function emojiCategory(title) {
		var cat = document.createElement("span");
		cat.className = "emoji-section";
		cat.innerText = title;
		return cat;
	}

	// element is input or textarea
	function setupForElement(element) {
		// 1. Create new div around element
		// 2. Move element into div
		// 3. Do the thing

		var div = element.parentElement;
		if (div.className != "emojipicker-container") {
			div = document.createElement("div");
			div.className = "emojipicker-container";
			element.parentElement.insertBefore(div, element);

			// move element
			div.appendChild(element);
			element.parentElement.removeChild(element);
		}

		var toggle = document.createElement("div");
		toggle.className = "emoji-toggle emoji-btn";
		toggle.innerText = "😃";

		div.appendChild(toggle);

		var ewindow = document.createElement("div");
		ewindow.className = "emoji-window emoji-window-closed";

		div.appendChild(ewindow);

		toggle.onclick = function() {
			if (ewindow.classList.contains("emoji-window-closed")) {
				ewindow.classList.remove("emoji-window-closed");
			} else {
				ewindow.classList.add("emoji-window-closed");
			}

			hideOnClickOutside(div, function() {
				ewindow.classList.add("emoji-window-closed");
			});
		};

		// Populate window
		for (const category of data) {
			ewindow.appendChild(emojiCategory(category["title"]));
			for (const em of category["set"].split(",")) {
				ewindow.appendChild(emojiBtn(em, element));
			}
		}
	}

	function setup() {
		var elems = document.getElementsByClassName("emojipicker-container");

		for (var i = 0; i < elems.length; i++) {
			var container = elems[i];
			var inputs = container.getElementsByTagName("input");
			for (var j = 0; j < inputs.length; j++) {
				var element = inputs[j];
				if (element.type != "text") continue;
				setupForElement(element);
			}

			var textareas = container.getElementsByTagName("textarea");
			for (var j = 0; j < textareas.length; j++) {
				var element = textareas[j];
				setupForElement(element);
			}
		}
	}

	// DATA
	var data = [
		{
			title: "People",
			set:
				"😀,😃,😄,😁,😆,😅,😂,🤣,😊,😇,🙂,🙃,😉,😌,😍,😘,😗,😙,😚,😋,😜,😝,😛,🤑,🤗,🤓,😎,🤡,🤠,😏,😒,😞,😔,😟,😕,🙁,☹️,😣,😖,😫,😩,😤,😠,😡,😶,😐,😑,😯,😦,😧,😮,😲,😵,😳,😱,😨,😰,😢,😥,🤤,😭,😓,😪,😴,🙄,🤔,🤥,😬,🤐,🤢,🤧,😷,🤒,🤕,😈,👿,👹,👺,💩,👻,💀,☠️,👽,👾,🤖,🎃,😺,😸,😹,😻,😼,😽,🙀,😿,😾,👐,🙌,👏,🙏,🤝,👍,👎,👊,✊,🤛,🤜,🤞,✌️,🤘,👌,👈,👉,👆,👇,☝️,✋,🤚,🖐,🖖,👋,🤙,💪,🖕,✍️,🤳,💅,💍,💄,💋,👄,👅,👂,👃,👣,👁,👀,🗣,👤,👥,👶,👦,👧,👨,👩,👱‍♀️,👱,👴,👵,👲,👳‍♀️,👳,👮‍♀️,👮,👷‍♀️,👷,💂‍♀️,💂,🕵️‍♀️,🕵️,👩‍⚕️,👨‍⚕️,👩‍🌾,👨‍🌾,👩‍🍳,👨‍🍳,👩‍🎓,👨‍🎓,👩‍🎤,👨‍🎤,👩‍🏫,👨‍🏫,👩‍🏭,👨‍🏭,👩‍💻,👨‍💻,👩‍💼,👨‍💼,👩‍🔧,👨‍🔧,👩‍🔬,👨‍🔬,👩‍🎨,👨‍🎨,👩‍🚒,👨‍🚒,👩‍✈️,👨‍✈️,👩‍🚀,👨‍🚀,👩‍⚖️,👨‍⚖️,🤶,🎅,👸,🤴,👰,🤵,👼,🤰,🙇‍♀️,🙇,💁,💁‍♂️,🙅,🙅‍♂️,🙆,🙆‍♂️,🙋,🙋‍♂️,🤦‍♀️,🤦‍♂️,🤷‍♀️,🤷‍♂️,🙎,🙎‍♂️,🙍,🙍‍♂️,💇,💇‍♂️,💆,💆‍♂️,🕴,💃,🕺,👯,👯‍♂️,🚶‍♀️,🚶,🏃‍♀️,🏃,👫,👭,👬,💑,👩‍❤️‍👩,👨‍❤️‍👨,💏,👩‍❤️‍💋‍👩,👨‍❤️‍💋‍👨,👪,👨‍👩‍👧,👨‍👩‍👧‍👦,👨‍👩‍👦‍👦,👨‍👩‍👧‍👧,👩‍👩‍👦,👩‍👩‍👧,👩‍👩‍👧‍👦,👩‍👩‍👦‍👦,👩‍👩‍👧‍👧,👨‍👨‍👦,👨‍👨‍👧,👨‍👨‍👧‍👦,👨‍👨‍👦‍👦,👨‍👨‍👧‍👧,👩‍👦,👩‍👧,👩‍👧‍👦,👩‍👦‍👦,👩‍👧‍👧,👨‍👦,👨‍👧,👨‍👧‍👦,👨‍👦‍👦,👨‍👧‍👧,👚,👕,👖,👔,👗,👙,👘,👠,👡,👢,👞,👟,👒,🎩,🎓,👑,⛑,🎒,👝,👛,👜,💼,👓,🕶,🌂,☂️"
		},
		{
			title: "Nature",
			set:
				"🐶,🐱,🐭,🐹,🐰,🦊,🐻,🐼,🐨,🐯,🦁,🐮,🐷,🐽,🐸,🐵,🙈,🙉,🙊,🐒,🐔,🐧,🐦,🐤,🐣,🐥,🦆,🦅,🦉,🦇,🐺,🐗,🐴,🦄,🐝,🐛,🦋,🐌,🐚,🐞,🐜,🕷,🕸,🐢,🐍,🦎,🦂,🦀,🦑,🐙,🦐,🐠,🐟,🐡,🐬,🦈,🐳,🐋,🐊,🐆,🐅,🐃,🐂,🐄,🦌,🐪,🐫,🐘,🦏,🦍,🐎,🐖,🐐,🐏,🐑,🐕,🐩,🐈,🐓,🦃,🕊,🐇,🐁,🐀,🐿,🐾,🐉,🐲,🌵,🎄,🌲,🌳,🌴,🌱,🌿,☘️,🍀,🎍,🎋,🍃,🍂,🍁,🍄,🌾,💐,🌷,🌹,🥀,🌻,🌼,🌸,🌺,🌎,🌍,🌏,🌕,🌖,🌗,🌘,🌑,🌒,🌓,🌔,🌚,🌝,🌞,🌛,🌜,🌙,💫,⭐️,🌟,✨,⚡️,🔥,💥,☄️,☀️,🌤,⛅️,🌥,🌦,🌈,☁️,🌧,⛈,🌩,🌨,☃️,⛄️,❄️,🌬,💨,🌪,🌫,🌊,💧,💦,☔️"
		},
		{
			title: "Foods",
			set:
				"🍏,🍎,🍐,🍊,🍋,🍌,🍉,🍇,🍓,🍈,🍒,🍑,🍍,🥝,🥑,🍅,🍆,🥒,🥕,🌽,🌶,🥔,🍠,🌰,🥜,🍯,🥐,🍞,🥖,🧀,🥚,🍳,🥓,🥞,🍤,🍗,🍖,🍕,🌭,🍔,🍟,🥙,🌮,🌯,🥗,🥘,🍝,🍜,🍲,🍥,🍣,🍱,🍛,🍚,🍙,🍘,🍢,🍡,🍧,🍨,🍦,🍰,🎂,🍮,🍭,🍬,🍫,🍿,🍩,🍪,🥛,🍼,☕️,🍵,🍶,🍺,🍻,🥂,🍷,🥃,🍸,🍹,🍾,🥄,🍴,🍽"
		},
		{
			title: "Activity",
			set:
				"⚽️,🏀,🏈,⚾️,🎾,🏐,🏉,🎱,🏓,🏸,🥅,🏒,🏑,🏏,⛳️,🏹,🎣,🥊,🥋,⛸,🎿,⛷,🏂,🏋️‍♀️,🏋️,🤺,🤼‍♀️,🤼‍♂️,🤸‍♀️,🤸‍♂️,⛹️‍♀️,⛹️,🤾‍♀️,🤾‍♂️,🏌️‍♀️,🏌️,🏄‍♀️,🏄,🏊‍♀️,🏊,🤽‍♀️,🤽‍♂️,🚣‍♀️,🚣,🏇,🚴‍♀️,🚴,🚵‍♀️,🚵,🎽,🏅,🎖,🥇,🥈,🥉,🏆,🏵,🎗,🎫,🎟,🎪,🤹‍♀️,🤹‍♂️,🎭,🎨,🎬,🎤,🎧,🎼,🎹,🥁,🎷,🎺,🎸,🎻,🎲,🎯,🎳,🎮,🎰"
		},
		{
			title: "Places",
			set:
				"🚗,🚕,🚙,🚌,🚎,🏎,🚓,🚑,🚒,🚐,🚚,🚛,🚜,🛴,🚲,🛵,🏍,🚨,🚔,🚍,🚘,🚖,🚡,🚠,🚟,🚃,🚋,🚞,🚝,🚄,🚅,🚈,🚂,🚆,🚇,🚊,🚉,🚁,🛩,✈️,🛫,🛬,🚀,🛰,💺,🛶,⛵️,🛥,🚤,🛳,⛴,🚢,⚓️,🚧,⛽️,🚏,🚦,🚥,🗺,🗿,🗽,⛲️,🗼,🏰,🏯,🏟,🎡,🎢,🎠,⛱,🏖,🏝,⛰,🏔,🗻,🌋,🏜,🏕,⛺️,🛤,🛣,🏗,🏭,🏠,🏡,🏘,🏚,🏢,🏬,🏣,🏤,🏥,🏦,🏨,🏪,🏫,🏩,💒,🏛,⛪️,🕌,🕍,🕋,⛩,🗾,🎑,🏞,🌅,🌄,🌠,🎇,🎆,🌇,🌆,🏙,🌃,🌌,🌉,🌁"
		},
		{
			title: "Objects",
			set:
				"⌚️,📱,📲,💻,⌨️,🖥,🖨,🖱,🖲,🕹,🗜,💽,💾,💿,📀,📼,📷,📸,📹,🎥,📽,🎞,📞,☎️,📟,📠,📺,📻,🎙,🎚,🎛,⏱,⏲,⏰,🕰,⌛️,⏳,📡,🔋,🔌,💡,🔦,🕯,🗑,🛢,💸,💵,💴,💶,💷,💰,💳,💎,⚖️,🔧,🔨,⚒,🛠,⛏,🔩,⚙️,⛓,🔫,💣,🔪,🗡,⚔️,🛡,🚬,⚰️,⚱️,🏺,🔮,📿,💈,⚗️,🔭,🔬,🕳,💊,💉,🌡,🚽,🚰,🚿,🛁,🛀,🛎,🔑,🗝,🚪,🛋,🛏,🛌,🖼,🛍,🛒,🎁,🎈,🎏,🎀,🎊,🎉,🎎,🏮,🎐,✉️,📩,📨,📧,💌,📥,📤,📦,🏷,📪,📫,📬,📭,📮,📯,📜,📃,📄,📑,📊,📈,📉,🗒,🗓,📆,📅,📇,🗃,🗳,🗄,📋,📁,📂,🗂,🗞,📰,📓,📔,📒,📕,📗,📘,📙,📚,📖,🔖,🔗,📎,🖇,📐,📏,📌,📍,✂️,🖊,🖋,✒️,🖌,🖍,📝,✏️,🔍,🔎,🔏,🔐,🔒,🔓"
		},
		{
			title: "Symbols",
			set:
				"❤️,💛,💚,💙,💜,🖤,💔,❣️,💕,💞,💓,💗,💖,💘,💝,💟,☮️,✝️,☪️,🕉,☸️,✡️,🔯,🕎,☯️,☦️,🛐,⛎,♈️,♉️,♊️,♋️,♌️,♍️,♎️,♏️,♐️,♑️,♒️,♓️,🆔,⚛️,🉑,☢️,☣️,📴,📳,🈶,🈚️,🈸,🈺,🈷️,✴️,🆚,💮,🉐,㊙️,㊗️,🈴,🈵,🈹,🈲,🅰️,🅱️,🆎,🆑,🅾️,🆘,❌,⭕️,🛑,⛔️,📛,🚫,💯,💢,♨️,🚷,🚯,🚳,🚱,🔞,📵,🚭,❗️,❕,❓,❔,‼️,⁉️,🔅,🔆,〽️,⚠️,🚸,🔱,⚜️,🔰,♻️,✅,🈯️,💹,❇️,✳️,❎,🌐,💠,Ⓜ️,🌀,💤,🏧,🚾,♿️,🅿️,🈳,🈂️,🛂,🛃,🛄,🛅,🚹,🚺,🚼,🚻,🚮,🎦,📶,🈁,🔣,ℹ️,🔤,🔡,🔠,🆖,🆗,🆙,🆒,🆕,🆓,0️⃣,1️⃣,2️⃣,3️⃣,4️⃣,5️⃣,6️⃣,7️⃣,8️⃣,9️⃣,🔟,🔢,#️⃣,*️⃣,▶️,⏸,⏯,⏹,⏺,⏭,⏮,⏩,⏪,⏫,⏬,◀️,🔼,🔽,➡️,⬅️,⬆️,⬇️,↗️,↘️,↙️,↖️,↕️,↔️,↪️,↩️,⤴️,⤵️,🔀,🔁,🔂,🔄,🔃,🎵,🎶,➕,➖,➗,✖️,💲,💱,™️,©️,®️,〰️,➰,➿,🔚,🔙,🔛,🔝,🔜,✔️,☑️,🔘,⚪️,⚫️,🔴,🔵,🔺,🔻,🔸,🔹,🔶,🔷,🔳,🔲,▪️,▫️,◾️,◽️,◼️,◻️,⬛️,⬜️,🔈,🔇,🔉,🔊,🔔,🔕,📣,📢,👁‍🗨,💬,💭,🗯,♠️,♣️,♥️,♦️,🃏,🎴,🀄️,🕐,🕑,🕒,🕓,🕔,🕕,🕖,🕗,🕘,🕙,🕚,🕛,🕜,🕝,🕞,🕟,🕠,🕡,🕢,🕣,🕤,🕥,🕦,🕧"
		},
		{
			title: "Flags",
			set:
				"🏳️,🏴,🏁,🚩,🏳️‍🌈,🇦🇫,🇦🇽,🇦🇱,🇩🇿,🇦🇸,🇦🇩,🇦🇴,🇦🇮,🇦🇶,🇦🇬,🇦🇷,🇦🇲,🇦🇼,🇦🇺,🇦🇹,🇦🇿,🇧🇸,🇧🇭,🇧🇩,🇧🇧,🇧🇾,🇧🇪,🇧🇿,🇧🇯,🇧🇲,🇧🇹,🇧🇴,🇧🇶,🇧🇦,🇧🇼,🇧🇷,🇮🇴,🇻🇬,🇧🇳,🇧🇬,🇧🇫,🇧🇮,🇨🇻,🇰🇭,🇨🇲,🇨🇦,🇮🇨,🇰🇾,🇨🇫,🇹🇩,🇨🇱,🇨🇳,🇨🇽,🇨🇨,🇨🇴,🇰🇲,🇨🇬,🇨🇩,🇨🇰,🇨🇷,🇨🇮,🇭🇷,🇨🇺,🇨🇼,🇨🇾,🇨🇿,🇩🇰,🇩🇯,🇩🇲,🇩🇴,🇪🇨,🇪🇬,🇸🇻,🇬🇶,🇪🇷,🇪🇪,🇪🇹,🇪🇺,🇫🇰,🇫🇴,🇫🇯,🇫🇮,🇫🇷,🇬🇫,🇵🇫,🇹🇫,🇬🇦,🇬🇲,🇬🇪,🇩🇪,🇬🇭,🇬🇮,🇬🇷,🇬🇱,🇬🇩,🇬🇵,🇬🇺,🇬🇹,🇬🇬,🇬🇳,🇬🇼,🇬🇾,🇭🇹,🇭🇳,🇭🇰,🇭🇺,🇮🇸,🇮🇳,🇮🇩,🇮🇷,🇮🇶,🇮🇪,🇮🇲,🇮🇱,🇮🇹,🇯🇲,🇯🇵,🎌,🇯🇪,🇯🇴,🇰🇿,🇰🇪,🇰🇮,🇽🇰,🇰🇼,🇰🇬,🇱🇦,🇱🇻,🇱🇧,🇱🇸,🇱🇷,🇱🇾,🇱🇮,🇱🇹,🇱🇺,🇲🇴,🇲🇰,🇲🇬,🇲🇼,🇲🇾,🇲🇻,🇲🇱,🇲🇹,🇲🇭,🇲🇶,🇲🇷,🇲🇺,🇾🇹,🇲🇽,🇫🇲,🇲🇩,🇲🇨,🇲🇳,🇲🇪,🇲🇸,🇲🇦,🇲🇿,🇲🇲,🇳🇦,🇳🇷,🇳🇵,🇳🇱,🇳🇨,🇳🇿,🇳🇮,🇳🇪,🇳🇬,🇳🇺,🇳🇫,🇲🇵,🇰🇵,🇳🇴,🇴🇲,🇵🇰,🇵🇼,🇵🇸,🇵🇦,🇵🇬,🇵🇾,🇵🇪,🇵🇭,🇵🇳,🇵🇱,🇵🇹,🇵🇷,🇶🇦,🇷🇪,🇷🇴,🇷🇺,🇷🇼,🇧🇱,🇸🇭,🇰🇳,🇱🇨,🇵🇲,🇻🇨,🇼🇸,🇸🇲,🇸🇹,🇸🇦,🇸🇳,🇷🇸,🇸🇨,🇸🇱,🇸🇬,🇸🇽,🇸🇰,🇸🇮,🇸🇧,🇸🇴,🇿🇦,🇬🇸,🇰🇷,🇸🇸,🇪🇸,🇱🇰,🇸🇩,🇸🇷,🇸🇿,🇸🇪,🇨🇭,🇸🇾,🇹🇼,🇹🇯,🇹🇿,🇹🇭,🇹🇱,🇹🇬,🇹🇰,🇹🇴,🇹🇹,🇹🇳,🇹🇷,🇹🇲,🇹🇨,🇹🇻,🇺🇬,🇺🇦,🇦🇪,🇬🇧,🇺🇸,🇻🇮,🇺🇾,🇺🇿,🇻🇺,🇻🇦,🇻🇪,🇻🇳,🇼🇫,🇪🇭,🇾🇪,🇿🇲,🇿🇼"
		}
	];

	if (options) {
		if (options.categories) {
			for (var k = 0; k < options.categories.length; k++) {
				if (options.categories[k]) {
					data[k].title = options.categories[k];
				}
			}
		}

		if (options.el) {
			setup(options.el);
		} else {
			setup();
		}
	} else {
		setup();
	}
}

emojipicker();