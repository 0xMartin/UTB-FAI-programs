/**
 * CodeWord - Pro znak uchovava informaci o jeho pravdepodobnosti vyskytu v textu a jeho kodove slovo
 */
public class CodeWord {

    //znak
    public Character character;

    //pravdepodobnost vyskytu
    public double probability;

    //kodove slovo znaku
    public String code = "";

    public CodeWord(Character character, double probability){
        this.character = character;
        this.probability = probability;
    }

}