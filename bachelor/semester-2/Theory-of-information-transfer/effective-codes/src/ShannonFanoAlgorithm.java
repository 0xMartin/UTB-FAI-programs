import java.util.*;

public class ShannonFanoAlgorithm extends EffectiveCodeAlghoritm{

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

        super.lastText = text;

        //znaky, ktere se nachazeji ve vstupnim textu
        Character[] chars = getIncludedChars(text);

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

        //kodova slova usporadaji sestupne podle pravdepodopnosti vyskytu jejich prirazenych znaku
        Arrays.sort(super.result, new Comparator<CodeWord>() {
            @Override
            public int compare(CodeWord n1, CodeWord n2) {
                return -Double.compare(n1.probability, n2.probability);
            }
        });

        //algoritmus pro vytvoreni efektivniho kodu Shannon Fanovou metodou
        alghoritm(super.result);

        return true;
    }

    private void alghoritm(CodeWord[] codeWords){
        if(codeWords.length <= 1){
            return;
        }

        //secte vsechny pravepodobnosti vsech znaku
        double totalPropbability = 0;
        for(CodeWord n : codeWords){
            totalPropbability += n.probability;
        }

        //najde delici index
        int splitIndex = 0;
        double minDiff = 1.0d;
        double sum = 0;
        for(int i = 0; i < codeWords.length; i++){
            sum += codeWords[i].probability;
            double diff = Math.abs(totalPropbability/2d - sum);
            if(diff <= minDiff) {
                minDiff = diff;
                splitIndex = i;
            }
        }

        //rozdeli pole na dve casti podle deliciho indexu
        CodeWord[] part1 = Arrays.copyOfRange(codeWords, 0, splitIndex+ 1);
        CodeWord[] part2 = Arrays.copyOfRange(codeWords, splitIndex+1, codeWords.length);

        //vsem znakum prvni casti pole priradi znak '1' do jejich kodoveho slova
        for(CodeWord n : part1){
            n.code += '1';
        }

        //vsem znakum prvni casti pole priradi znak '0' do jejich kodoveho slova
        for(CodeWord n : part2){
            n.code += '0';
        }

        //rekurzivni opakovani
        alghoritm(part1);
        alghoritm(part2);
    }

}
