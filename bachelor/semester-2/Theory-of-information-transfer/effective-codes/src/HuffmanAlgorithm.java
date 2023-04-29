import java.util.Arrays;
import java.util.Comparator;

public class HuffmanAlgorithm extends EffectiveCodeAlghoritm {

    @Override
    public boolean createCode(String text) {
        if(text.equals(super.lastText)){
            return false;
        }

        super.lastText = text;

        if(text.length() == 0) {
            super.result = null;
            return true;
        }

        //znaky, ktere se nachazeji ve vstupnim textu
        Character[] chars = EffectiveCodeAlghoritm.getIncludedChars(text);

        //pokud kod obsahuje jen jeden znak
        if(chars.length == 1) {
            super.result = new CodeWord[]{new CodeWord(text.charAt(0), 1.0)};
            super.result[0].code = "0";
            return true;
        }

        //vypocet pravdepodobnosti vyskytu jednotlivych znaku
        super.result = new CodeWord[chars.length];
        for(int i = 0; i < chars.length; i++){
            super.result[i] = new CodeWord(
                    chars[i],
                    (double)countChars(text, chars[i])/text.length()
            );
        }

        //algoritmus pro vytvoreni efektivniho kodu Huffmanovou metodou
        alghoritm(super.result);

        return true;
    }

    private void alghoritm(CodeWord[] codeWords){
        if(codeWords.length <= 1){
            return;
        }

        //kodova slova usporadaji sestupne podle pravdepodopnosti vyskytu jejich prirazenych znaku
        Arrays.sort(codeWords, new Comparator<CodeWord>() {
            @Override
            public int compare(CodeWord n1, CodeWord n2) {
                return -Double.compare(n1.probability, n2.probability);
            }
        });

        //kodova slova s druhou nejmensi a nejmensi pravdepodobnosti vyskytu
        CodeWord higher = codeWords[codeWords.length-2];
        CodeWord lower = codeWords[codeWords.length-1];

        if(codeWords.length > 2) {
            //potomek dvou kodovych slov s nejmensi pravdepodobnosti
            CodeWord child = new CodeWord('#', higher.probability + lower.probability);

            //vytvoreni kopie pole, ve kterem jsou dve kodove slova s nejmensi pravdepodobnosti nahrazeny jejich potomkem
            CodeWord[] reduced = new CodeWord[codeWords.length - 1];
            System.arraycopy(codeWords, 0, reduced, 0, codeWords.length - 2);
            reduced[codeWords.length - 2] = child;

            //rekurze
            alghoritm(reduced);

            //kodove znaky potomka jsou prirazeny rodicovskym kodovym slovum
            higher.code += child.code;
            lower.code += child.code;
        }

        //priradeni kodovych znaku
        higher.code += '0';
        lower.code += '1';
    }

}
