import java.util.ArrayList;
import java.util.List;

public abstract class EffectiveCodeAlghoritm {

    //posledni kodovana zprava
    protected String lastText = "";

    //kod
    protected CodeWord[] result = null;

    /**
     * Vytvori efektivni kod pro vstupni text
     * @param text Vstupni text
     * @return True: kodovani probehlo uspesne, False: probehlo neuspesne
     */
    public abstract boolean createCode(String text);

    /**
     * Vrati kod
     * @return Kod
     */
    public CodeWord[] getCode(){
        return this.result;
    }

    /**
     * Navrati zakodovany text
     * @return Zakodovany text
     */
    public String getCodedText(){
        StringBuilder ret = new StringBuilder();

        for(Character c : this.lastText.toCharArray()){
            for(CodeWord n : this.result) {
                if(n.character == c){
                    ret.append(n.code+" ");
                    break;
                }
            }
        }

        return ret.toString();
    }

    /**
     * Vypocita prumernou delku kodu
     * @return Prumerna delka
     */
    public double getAvgCodeLength(){
        if(this.result == null){
            return 0d;
        }

        double avgLength = 0;
        for(CodeWord n : this.result) {
            avgLength += n.code.length() * n.probability;
        }

        return avgLength;
    }

    /**
     * Vypocita efektivitu kodu
     * @return Efektivita
     */
    public double getEffectivity(){
        if(this.result == null){
            return 0d;
        }

        //vypocet entropie
        double entropy = 0;
        for(CodeWord n : this.result) {
            entropy += n.probability * Math.log(n.probability) / Math.log(2);
        }

        double avg = getAvgCodeLength();

        //vypocet efektivity
        return avg == 0 ? 0 : entropy / -avg;
    }

    /**
     * Navrati pole se znaky, ktere se v tomto poli nachazeji (bez opakovani znaku)
     * @param text Vstupni text
     * @return Pole se znaky, ktere se v tomto poli nachazi
     */
    protected static Character[] getIncludedChars(String text){
        List<Character> charList = new ArrayList<Character>();
        text.chars().forEach((c1)->{
            if(charList.stream().allMatch(c2 -> (c1 != c2))) {
                charList.add((char) c1);
            }
        });

        Character[] charArray = new Character[charList.size()];
        charArray = charList.toArray(charArray);
        return charArray;
    }

    /**
     * Spocita kolikrat se dany znak nachazi ve vstupnim textu
     * @param text Vstupni text
     * @param character Znaky, ktery budeme pocitat
     * @return Pocet znak v textu
     */
    protected static int countChars(String text, char character){
        int count = 0;
        for(int i = 0; i < text.length(); ++i){
            count = text.charAt(i) == character ? count + 1 : count;
        }
        return count;
    }

}
