jQuery.validator.addMethod('unique',
    function (value, element, params) {
        var min = params[1];

        if (value) {

            // A map (in JavaScript, an object) for the character=>count mappings
            var counts = {};

            // Misc vars
            var ch, index, len, count;

            // Loop through the string...
            for (index = 0, len = value.length; index < len; ++index) {
                // Get this character
                ch = value.charAt(index); // Not all engines support [] on strings

                // Get the count for it, if we have one; we'll get `undefined` if we
                // don't know this character yet
                count = counts[ch];

                // If we have one, store that count plus one; if not, store one
                // We can rely on `count` being falsey if we haven't seen it before,
                // because we never store falsey numbers in the `counts` object.
                counts[ch] = count ? count + 1 : 1;
            }

            var numberOfUniqueCharacters = 0;
            for (x in counts) {
                numberOfUniqueCharacters += 1;
            }

            if (numberOfUniqueCharacters >= min) {
                return true;
            }
        }

        return false;
    });

jQuery.validator.unobtrusive.adapters.add('filecontent', ['min'],
    function (options) {
        var element = $(options.form).find('#password')[0];
        options.rules['filecontent'] = [element, parseInt(options.params['min'])];
        options.messages['filecontent'] = options.message;
    });